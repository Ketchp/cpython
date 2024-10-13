import ast
import contextlib
import re
from abc import abstractmethod
from functools import wraps
from typing import (
    IO,
    AbstractSet,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    ParamSpec,
    Set,
    Text,
    Tuple,
    TypeVar,
    Union,
    Callable,
)

from pegen import sccutils
from pegen.grammar import (
    Alt,
    Cut,
    Forced,
    Gather,
    Grammar,
    GrammarError,
    GrammarVisitor,
    Group,
    Lookahead,
    NamedItem,
    NameLeaf,
    Opt,
    Repeat,
    Repeat0,
    Repeat1,
    Rhs,
    Rule,
    StringLeaf,
    Item,
)


param_T = ParamSpec("param_T")
return_T = TypeVar("return_T")


class RuleCollectorVisitor(GrammarVisitor):
    """Visitor that invokes a provided callmaker visitor with just the NamedItem nodes"""

    def __init__(self, rules: Dict[str, Rule], callmakervisitor: GrammarVisitor) -> None:
        self.rulses = rules
        self.callmaker = callmakervisitor

    def visit_Rule(self, rule: Rule) -> None:
        self.visit(rule.flatten())

    def visit_NamedItem(self, item: NamedItem) -> None:
        self.callmaker.visit(item)


class KeywordCollectorVisitor(GrammarVisitor):
    """Visitor that collects all the keywods and soft keywords in the Grammar"""

    def __init__(self, gen: "ParserGenerator", keywords: Dict[str, int], soft_keywords: Set[str]):
        self.generator = gen
        self.keywords = keywords
        self.soft_keywords = soft_keywords

    def visit_StringLeaf(self, node: StringLeaf) -> None:
        val = ast.literal_eval(node.value)
        if re.match(r"[a-zA-Z_]\w*\Z", val):  # This is a keyword
            if node.value.endswith("'"):
                if val not in self.keywords:
                    self.keywords[val] = self.generator.keyword_type()
            else:
                return self.soft_keywords.add(node.value)


class RuleCheckingVisitor(GrammarVisitor):
    def __init__(self, rules: Dict[str, Rule], tokens: Set[str]):
        self.rules = rules
        self.tokens = tokens

    def visit_NameLeaf(self, node: NameLeaf) -> None:
        if node.value not in self.rules and node.value not in self.tokens:
            raise GrammarError(f"Dangling reference to rule {node.value!r}")

    def visit_NamedItem(self, node: NamedItem) -> None:
        if node.name and node.name.startswith("_"):
            raise GrammarError(f"Variable names cannot start with underscore: '{node.name}'")
        self.visit(node.item)


class TransformerVisitor(GrammarVisitor):
    """Transforms repeat/gather/group rules into simpler ones."""
    def __init__(self, rules: Dict[str, Rule]):
        self.rules = rules
        self._artificial_rule_cache: Dict[str, NameLeaf] = {}
        self._counter = 0

    def trivial_visit_wrapper(self, attr: str = '') -> Callable[[Item], Item]:
        def visit_function(node: Item) -> Item:
            if attr:
                setattr(node, attr, self.visit(getattr(node, attr)))
            return node
        return visit_function

    def __getattr__(self, key: str) -> Callable[[Item], Item]:
        leaf_visit = self.trivial_visit_wrapper()
        item_visit = self.trivial_visit_wrapper('item')
        node_visit = self.trivial_visit_wrapper('node')

        mapping = {
            'Cut': leaf_visit,
            'NameLeaf': leaf_visit,
            'StringLeaf': leaf_visit,

            'Opt': node_visit,
            'Forced': node_visit,
            'PositiveLookahead': node_visit,
            'NegativeLookahead': node_visit,

            'NamedItem': item_visit,
        }

        if key.startswith('visit_'):
            key = key.removeprefix('visit_')
            if key in mapping:
                return mapping[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def visit_Rule(self, rule: Rule) -> Rule:
        # flatten used for Groups as single Rule alt
        rule.rhs = self.visit(rule.flatten(), is_rule_rhs=True)
        return rule

    def visit_Alt(self, node: Alt) -> Alt:
        node.items = [self.visit(item) for item in node.items]
        return node

    def visit_Rhs(self, rhs: Rhs, is_rule_rhs: bool = False) -> Union[Rhs | NameLeaf]:
        if is_rule_rhs or rhs.can_be_inlined:
            rhs.alts = [self.visit(alt) for alt in rhs.alts]
            return rhs
        return self._artificial_rule_from_rhs(rhs)

    def visit_Repeat0(self, node: Repeat0) -> NameLeaf:
        return self._artificial_rule_from_repeat(node, is_repeat1=False)

    def visit_Repeat1(self, node: Repeat0) -> NameLeaf:
        return self._artificial_rule_from_repeat(node, is_repeat1=True)

    def visit_Gather(self, node: Gather) -> NameLeaf:
        return self._artificial_rule_from_gather(node)

    def visit_Group(self, group: Group) -> NameLeaf:
        return self.visit(group.rhs)   # groups always create new rule

    @staticmethod
    def _rule_cached(func: Callable[param_T, return_T]) -> Callable[param_T, return_T]:
        """Make sure artificial rules are not duplicated."""
        @wraps(func)
        def inner(self, node, *args, **kwargs) -> return_T:
            key = (type(node), str(node))

            if key in self._artificial_rule_cache:
                return self._artificial_rule_cache[key]

            ret = func(self, node, *args, **kwargs)
            self._artificial_rule_cache[key] = ret
            return ret
        return inner

    @_rule_cached
    def _artificial_rule_from_rhs(self, rhs: Rhs) -> NameLeaf:
        self._counter += 1
        name = f"_tmp_{self._counter}"  # TODO: Pick a nicer name.
        self.rules[name] = Rule(name, None, rhs)
        return NameLeaf(name, comment=str(rhs))

    @_rule_cached
    def _artificial_rule_from_repeat(self, node: Repeat, is_repeat1: bool) -> NameLeaf:
        self._counter += 1
        if is_repeat1:
            prefix = "_loop1_"
        else:
            prefix = "_loop0_"
        name = f"{prefix}{self._counter}"
        self.rules[name] = Rule(
            name,
            None,
            Rhs([Alt([NamedItem(None, node.node)])]),
        )
        return NameLeaf(name, comment=str(node))

    @_rule_cached
    def _artificial_rule_from_gather(self, node: Gather) -> NameLeaf:
        self._counter += 1
        extra_function_name = f"_loop0_{self._counter}"
        extra_function_alt = Alt(
            [NamedItem(None, node.separator), NamedItem("elem", node.node)],
            action="elem",
        )
        self.rules[extra_function_name] = Rule(
            extra_function_name,
            None,
            Rhs([extra_function_alt]),
        )

        self._counter += 1
        name = f"_gather_{self._counter}"
        alt = Alt(
            [NamedItem("elem", node.node), NamedItem("seq", NameLeaf(extra_function_name))],
        )
        self.rules[name] = Rule(
            name,
            None,
            Rhs([alt]),
        )
        return NameLeaf(name, comment=str(node))


class ParserGenerator:
    callmakervisitor: GrammarVisitor

    def __init__(self, grammar: Grammar, tokens: Set[str], file: Optional[IO[Text]]):
        self.grammar = grammar
        self.tokens = tokens
        self.keywords: Dict[str, int] = {}
        self.soft_keywords: Set[str] = set()
        self.rules = grammar.rules
        self.validate_rule_names()
        if "trailer" not in grammar.metas and "start" not in self.rules:
            raise GrammarError("Grammar without a trailer must have a 'start' rule")
        checker = RuleCheckingVisitor(self.rules, self.tokens)
        for rule in self.rules.values():
            checker.visit(rule)
        self.file = file
        self.level = 0
        self.first_graph, self.first_sccs = compute_left_recursives(self.rules)
        self.counter = 0  # For name_rule()/name_loop()
        self.keyword_counter = 499  # For keyword_type()
        self.all_rules: Dict[str, Rule] = self.rules.copy()  # Rules + temporal rules
        self._local_variable_stack: List[List[str]] = []

    def validate_rule_names(self) -> None:
        for rule in self.rules:
            if rule.startswith("_"):
                raise GrammarError(f"Rule names cannot start with underscore: '{rule}'")

    @contextlib.contextmanager
    def local_variable_context(self) -> Iterator[None]:
        self._local_variable_stack.append([])
        yield
        self._local_variable_stack.pop()

    @property
    def local_variable_names(self) -> List[str]:
        return self._local_variable_stack[-1]

    @abstractmethod
    def generate(self, filename: str) -> None:
        raise NotImplementedError

    @contextlib.contextmanager
    def indent(self) -> Iterator[None]:
        self.level += 1
        try:
            yield
        finally:
            self.level -= 1

    def print(self, *args: object) -> None:
        if not args:
            print(file=self.file)
        else:
            print("    " * self.level, end="", file=self.file)
            print(*args, file=self.file)

    def printblock(self, lines: str) -> None:
        for line in lines.splitlines():
            self.print(line)

    def collect_rules(self) -> None:
        keyword_collector = KeywordCollectorVisitor(self, self.keywords, self.soft_keywords)
        for rule in self.all_rules.values():
            keyword_collector.visit(rule)

        transformer = TransformerVisitor(self.all_rules)
        rule_collector = RuleCollectorVisitor(self.all_rules, self.callmakervisitor)
        done: Set[str] = set()
        while True:
            computed_rules = list(self.all_rules)
            todo = [i for i in computed_rules if i not in done]
            if not todo:
                break
            done = set(self.all_rules)
            for rulename in todo:
                transformer.visit(self.all_rules[rulename])
                rule_collector.visit(self.all_rules[rulename])

    def keyword_type(self) -> int:
        self.keyword_counter += 1
        return self.keyword_counter

    def dedupe(self, name: str) -> str:
        origname = name
        counter = 0
        while name in self.local_variable_names:
            counter += 1
            name = f"{origname}_{counter}"
        self.local_variable_names.append(name)
        return name


class NullableVisitor(GrammarVisitor):
    def __init__(self, rules: Dict[str, Rule]) -> None:
        self.rules = rules
        self.visited: Set[Any] = set()
        self.nullables: Set[Union[Rule, NamedItem]] = set()

    def visit_Rule(self, rule: Rule) -> bool:
        if rule in self.visited:
            return False
        self.visited.add(rule)
        if self.visit(rule.rhs):
            self.nullables.add(rule)
        return rule in self.nullables

    def visit_Rhs(self, rhs: Rhs) -> bool:
        for alt in rhs.alts:
            if self.visit(alt):
                return True
        return False

    def visit_Alt(self, alt: Alt) -> bool:
        for item in alt.items:
            if not self.visit(item):
                return False
        return True

    def visit_Forced(self, force: Forced) -> bool:
        return True

    def visit_LookAhead(self, lookahead: Lookahead) -> bool:
        return True

    def visit_Opt(self, opt: Opt) -> bool:
        return True

    def visit_Repeat0(self, repeat: Repeat0) -> bool:
        return True

    def visit_Repeat1(self, repeat: Repeat1) -> bool:
        return False

    def visit_Gather(self, gather: Gather) -> bool:
        return False

    def visit_Cut(self, cut: Cut) -> bool:
        return False

    def visit_Group(self, group: Group) -> bool:
        return self.visit(group.rhs)

    def visit_NamedItem(self, item: NamedItem) -> bool:
        if self.visit(item.item):
            self.nullables.add(item)
        return item in self.nullables

    def visit_NameLeaf(self, node: NameLeaf) -> bool:
        if node.value in self.rules:
            return self.visit(self.rules[node.value])
        # Token or unknown; never empty.
        return False

    def visit_StringLeaf(self, node: StringLeaf) -> bool:
        # The string token '' is considered empty.
        return not node.value


def compute_nullables(rules: Dict[str, Rule]) -> Set[Any]:
    """Compute which rules in a grammar are nullable.

    Thanks to TatSu (tatsu/leftrec.py) for inspiration.
    """
    nullable_visitor = NullableVisitor(rules)
    for rule in rules.values():
        nullable_visitor.visit(rule)
    return nullable_visitor.nullables


class InitialNamesVisitor(GrammarVisitor):
    def __init__(self, rules: Dict[str, Rule]) -> None:
        self.rules = rules
        self.nullables = compute_nullables(rules)

    def generic_visit(self, node: Iterable[Any], *args: Any, **kwargs: Any) -> Set[Any]:
        names: Set[str] = set()
        for value in node:
            names |= self.visit(value, *args, **kwargs)
        return names

    def visit_Alt(self, alt: Alt) -> Set[Any]:
        names: Set[str] = set()
        for item in alt.items:
            names |= self.visit(item)
            if item not in self.nullables:
                break
        return names

    def visit_Forced(self, force: Forced) -> Set[Any]:
        return set()

    def visit_LookAhead(self, lookahead: Lookahead) -> Set[Any]:
        return set()

    def visit_Cut(self, cut: Cut) -> Set[Any]:
        return set()

    def visit_NameLeaf(self, node: NameLeaf) -> Set[Any]:
        return {node.value}

    def visit_StringLeaf(self, node: StringLeaf) -> Set[Any]:
        return set()


def compute_left_recursives(
    rules: Dict[str, Rule]
) -> Tuple[Dict[str, AbstractSet[str]], List[AbstractSet[str]]]:
    graph = make_first_graph(rules)
    sccs = list(sccutils.strongly_connected_components(graph.keys(), graph))
    for scc in sccs:
        if len(scc) > 1:
            for name in scc:
                rules[name].left_recursive = True
            # Try to find a leader such that all cycles go through it.
            leaders = set(scc)
            for start in scc:
                for cycle in sccutils.find_cycles_in_scc(graph, scc, start):
                    # print("Cycle:", " -> ".join(cycle))
                    leaders -= scc - set(cycle)
                    if not leaders:
                        raise ValueError(
                            f"SCC {scc} has no leadership candidate (no element is included in all cycles)"
                        )
            # print("Leaders:", leaders)
            leader = min(leaders)  # Pick an arbitrary leader from the candidates.
            rules[leader].leader = True
        else:
            name = min(scc)  # The only element.
            if name in graph[name]:
                rules[name].left_recursive = True
                rules[name].leader = True
    return graph, sccs


def make_first_graph(rules: Dict[str, Rule]) -> Dict[str, AbstractSet[str]]:
    """Compute the graph of left-invocations.

    There's an edge from A to B if A may invoke B at its initial
    position.

    Note that this requires the nullable flags to have been computed.
    """
    initial_name_visitor = InitialNamesVisitor(rules)
    graph = {}
    vertices: Set[str] = set()
    for rulename, rhs in rules.items():
        graph[rulename] = names = initial_name_visitor.visit(rhs)
        vertices |= names
    for vertex in vertices:
        graph.setdefault(vertex, set())
    return graph
