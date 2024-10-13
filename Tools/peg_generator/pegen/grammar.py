from __future__ import annotations

from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)


class GrammarError(Exception):
    pass


class GrammarVisitor:
    def visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        """Called if no explicit visitor function exists for a node."""
        for value in node:
            self.visit(value, *args, **kwargs)


class Grammar:
    def __init__(self, rules: Iterable[Rule], metas: Iterable[Tuple[str, Optional[str]]]):
        # Check if there are repeated rules in "rules"
        all_rules = {}
        for rule in rules:
            if rule.name in all_rules:
                raise GrammarError(f"Repeated rule {rule.name!r}")
            all_rules[rule.name] = rule
        self.rules = all_rules
        self.metas = dict(metas)

    def __str__(self) -> str:
        return "\n".join(str(rule) for name, rule in self.rules.items())

    def __repr__(self) -> str:
        return "\n".join((
            "Grammar(",
            "  [",
            *(f"    {repr(rule)}," for rule in self.rules.values()),
            "  ],",
            f"  {repr(list(self.metas.items()))}",
            ")",
        ))

    def __iter__(self) -> Iterator[Rule]:
        yield from self.rules.values()


# Global flag whether we want actions in __str__() -- default off.
SIMPLE_STR = True


class InitComment:
    def __init__(self, comment: Optional[str] = None):
        self.comment = comment or self._create_comment()

    def __str__(self) -> str:
        return self.comment

    def _create_comment(self) -> str:
        raise NotImplementedError("Subclass must implement _create_comment method")



class Rule(InitComment):
    def __init__(self, name: str, type: Optional[str], rhs: Rhs, memo: Optional[object] = None):
        self.name = name
        self.type = type
        self.rhs = rhs
        self.memo = bool(memo)
        self.left_recursive = False
        self.leader = False
        super().__init__()

    def is_loop(self) -> bool:
        return self.name.startswith("_loop")

    def is_gather(self) -> bool:
        return self.name.startswith("_gather")

    def _create_comment(self) -> str:
        if SIMPLE_STR or self.type is None:
            res = f"{self.name}: {self.rhs}"
        else:
            res = f"{self.name}[{self.type}]: {self.rhs}"
        if len(res) < 88:
            return res
        lines = [res.split(":")[0] + ":"]
        lines += [f"    | {alt}" for alt in self.rhs.alts]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Rule({self.name!r}, {self.type!r}, {self.rhs!r})"

    def __iter__(self) -> Iterator[Rhs]:
        yield self.rhs

    def flatten(self) -> Rhs:
        # If it's a single parenthesized group, flatten it.
        rhs = self.rhs
        if (
            not self.is_loop()
            and len(rhs.alts) == 1
            and len(rhs.alts[0].items) == 1
            and isinstance(rhs.alts[0].items[0].item, Group)
        ):
            rhs = rhs.alts[0].items[0].item.rhs
        return rhs


class Leaf(InitComment):
    def __init__(self, value: str, comment: Optional[str] = None):
        self.value = value
        super().__init__(comment)

    def _create_comment(self) -> str:
        return self.value

    def __iter__(self) -> Iterable[str]:
        yield from ()


class NameLeaf(Leaf):
    """The value is the name."""

    def _create_comment(self) -> str:
        if self.value == "ENDMARKER":
            return "$"
        return super()._create_comment()

    def __repr__(self) -> str:
        return f"NameLeaf({self.value!r})"


class StringLeaf(Leaf):
    """The value is a string literal, including quotes."""

    def __repr__(self) -> str:
        return f"StringLeaf({self.value!r})"


class Rhs(InitComment):
    def __init__(self, alts: List[Alt]):
        self.alts = alts
        self.memo: Optional[Tuple[Optional[str], str]] = None
        super().__init__()

    def _create_comment(self) -> str:
        return " | ".join(str(alt) for alt in self.alts)

    def __repr__(self) -> str:
        return f"Rhs({self.alts!r})"

    def __iter__(self) -> Iterator[Alt]:
        yield from self.alts

    @property
    def can_be_inlined(self) -> bool:
        if len(self.alts) != 1 or len(self.alts[0].items) != 1:
            return False
        # If the alternative has an action we cannot inline
        return self.alts[0].action is None


class Alt(InitComment):
    def __init__(self, items: List[NamedItem], *, action: Optional[str] = None):
        self.items = items
        self.action = action
        super().__init__()

    def _create_comment(self) -> str:
        core = " ".join(str(item) for item in self.items)
        if not SIMPLE_STR and self.action:
            return f"{core} {{ {self.action} }}"
        else:
            return core

    def __repr__(self) -> str:
        args = [repr(self.items)]
        if self.action:
            args.append(f"action={self.action!r}")
        return f"Alt({', '.join(args)})"

    def __iter__(self) -> Iterator[NamedItem]:
        yield from self.items


class NamedItem(InitComment):
    def __init__(self, name: Optional[str], item: Item, type: Optional[str] = None):
        self.name = name
        self.item = item
        self.type = type
        super().__init__()

    def _create_comment(self) -> str:
        if not SIMPLE_STR and self.name:
            return f"{self.name}={self.item}"
        else:
            return str(self.item)

    def __repr__(self) -> str:
        return f"NamedItem({self.name!r}, {self.item!r})"

    def __iter__(self) -> Iterator[Item]:
        yield self.item


class Forced(InitComment):
    def __init__(self, node: Plain):
        self.node = node
        super().__init__()

    def _create_comment(self) -> str:
        return f"&&{self.node}"

    def __iter__(self) -> Iterator[Plain]:
        yield self.node


class Lookahead(InitComment):
    def __init__(self, node: Plain, sign: str):
        self.node = node
        self.sign = sign
        super().__init__()

    def _create_comment(self) -> str:
        return f"{self.sign}{self.node}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node!r})"

    def __iter__(self) -> Iterator[Plain]:
        yield self.node


class PositiveLookahead(Lookahead):
    def __init__(self, node: Plain):
        super().__init__(node, "&")


class NegativeLookahead(Lookahead):
    def __init__(self, node: Plain):
        super().__init__(node, "!")


class Opt(InitComment):
    def __init__(self, node: Item):
        self.node = node
        super().__init__()

    def _create_comment(self) -> str:
        s = str(self.node)
        if isinstance(s, Group):
            s = s[1:-1]  # strip '(' and ')'
        return f"[{s}]"

    def __repr__(self) -> str:
        return f"Opt({self.node!r})"

    def __iter__(self) -> Iterator[Item]:
        yield self.node


class Repeat(InitComment):
    """Shared base class for x* and x+."""

    def __init__(self, node: Plain):
        self.node = node
        self.memo: Optional[Tuple[Optional[str], str]] = None
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.node!r})"

    def __iter__(self) -> Iterator[Plain]:
        yield self.node


class Repeat0(Repeat):
    def _create_comment(self) -> str:
        return f"{self.node}*"


class Repeat1(Repeat):
    def _create_comment(self) -> str:
        return f"{self.node}+"


class Gather(Repeat):
    def __init__(self, separator: Plain, node: Plain):
        self.separator = separator
        super().__init__(node)

    def _create_comment(self) -> str:
        return f"{self.separator}.{self.node}+"

    def __repr__(self) -> str:
        return f"Gather({self.separator!r}, {self.node!r})"


class Group(InitComment):
    def __init__(self, rhs: Rhs):
        self.rhs = rhs
        super().__init__()

    def _create_comment(self) -> str:
        return f"({self.rhs})"

    def __repr__(self) -> str:
        return f"Group({self.rhs!r})"

    def __iter__(self) -> Iterator[Rhs]:
        yield self.rhs


class Cut(InitComment):
    def _create_comment(self) -> str:
        return "~"

    def __repr__(self) -> str:
        return f"Cut()"

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        yield from ()


Plain = Union[Leaf, Group]
Item = Union[Plain, Opt, Repeat, Forced, Lookahead, Rhs, Cut]
RuleName = Tuple[str, Optional[str]]
MetaTuple = Tuple[str, Optional[str]]
MetaList = List[MetaTuple]
RuleList = List[Rule]
NamedItemList = List[NamedItem]
LookaheadOrCut = Union[Lookahead, Cut]
