//! The surface language: sorts, values, and the `query` form.

use std::fmt;
use std::str::FromStr;

use crate::nouns::{Context, Domino, Seat, Team};
use crate::{Error, Result, error};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Sort {
    Context,
    Chair,
    Domino,
    /// Literal-only: a number cannot be a role.
    Number,
    /// Literal-only: a team cannot be a role.
    Team,
}

impl Sort {
    pub fn is_role(self) -> bool {
        matches!(self, Self::Context | Self::Chair | Self::Domino)
    }

    pub(crate) fn values(self) -> Vec<Value> {
        match self {
            Self::Context => Context::ALL.into_iter().map(Value::Context).collect(),
            Self::Chair => Seat::ALL.into_iter().map(Value::Chair).collect(),
            Self::Domino => Domino::ALL.into_iter().map(Value::Domino).collect(),
            Self::Number | Self::Team => Vec::new(),
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Context => "context",
            Self::Chair => "chair",
            Self::Domino => "domino",
            Self::Number => "number",
            Self::Team => "team",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Value {
    Context(Context),
    Chair(Seat),
    Domino(Domino),
    Number(u32),
    Team(Team),
}

impl Value {
    pub fn sort(self) -> Sort {
        match self {
            Self::Context(_) => Sort::Context,
            Self::Chair(_) => Sort::Chair,
            Self::Domino(_) => Sort::Domino,
            Self::Number(_) => Sort::Number,
            Self::Team(_) => Sort::Team,
        }
    }

    /// Literal spellings: `6-4`, `S0`..`S3`, `q0`..`q6`, `q*`, `T0`, `T1`, and
    /// nonnegative integers.
    pub(crate) fn literal(s: &str) -> Option<Self> {
        if let Ok(d) = s.parse() {
            return Some(Self::Domino(d));
        }
        if let Ok(q) = s.parse() {
            return Some(Self::Context(q));
        }
        if let Some(n) = s.strip_prefix('S') {
            return n.parse().ok().and_then(Seat::from_index).map(Self::Chair);
        }
        match s {
            "T0" => Some(Self::Team(Team::T0)),
            "T1" => Some(Self::Team(Team::T1)),
            _ => s.parse().ok().map(Self::Number),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context(x) => x.fmt(f),
            Self::Chair(x) => x.fmt(f),
            Self::Domino(x) => x.fmt(f),
            Self::Number(x) => x.fmt(f),
            Self::Team(x) => x.fmt(f),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub sort: Sort,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    Role(String),
    Literal(Value),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atom {
    pub predicate: String,
    pub args: Vec<Term>,
    /// An undefined atom satisfies neither itself nor its negation.
    pub negated: bool,
}

/// One conjunctive branch of a query, or the body of a `none`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    /// Same-sort equality groups over the roles declared with this body,
    /// transitively closed when compiled. Roles of one sort declared together
    /// and in different groups denote DISTINCT values.
    pub equal: Vec<Vec<String>>,
    pub atoms: Vec<Atom>,
    pub nones: Vec<NoneClause>,
}

/// `(none (roles ...) clause ...)`: no binding of its own roles satisfies its
/// body, which is read once the enclosing roles are bound. An assignment
/// whose body is undefined (neither satisfied nor refuted) leaves the clause
/// undefined, so it never manufactures a fact from missing context.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoneClause {
    pub roles: Vec<Role>,
    pub body: Pattern,
}

/// A relation: the set of output tuples for which some pattern has a
/// satisfying assignment of the remaining (existential) roles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Query {
    pub roles: Vec<Role>,
    pub outputs: Vec<String>,
    pub patterns: Vec<Pattern>,
}

pub(crate) fn valid_name(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// A parsed s-expression, shared by the query and policy readers.
#[derive(Clone, Debug)]
pub(crate) enum Expr {
    Word(String),
    List(Vec<Expr>),
}

impl Expr {
    /// Read exactly one form from `source`, with `;` comments, at most
    /// `max_bytes` long and `max_depth` deep.
    pub(crate) fn read(source: &str, what: &str, max_bytes: usize, max_depth: usize) -> Result<Expr> {
        if source.len() > max_bytes {
            return Err(error(format!("{what} exceeds {max_bytes} bytes")));
        }
        let source: String = source.lines().map(|l| l.split(';').next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let spaced = source.replace('(', " ( ").replace(')', " ) ");
        let tokens: Vec<&str> = spaced.split_whitespace().collect();
        let mut at = 0;
        let tree = Self::read_at(&tokens, &mut at, 0, what, max_depth)?;
        if at != tokens.len() {
            return Err(error(format!("trailing forms after the {what}")));
        }
        Ok(tree)
    }

    fn read_at(tokens: &[&str], at: &mut usize, depth: usize, what: &str, max_depth: usize) -> Result<Expr> {
        if depth > max_depth {
            return Err(error(format!("{what} nesting exceeds {max_depth}")));
        }
        let token = *tokens.get(*at).ok_or_else(|| error(format!("unexpected end of {what}")))?;
        *at += 1;
        match token {
            ")" => Err(error("unexpected closing parenthesis")),
            "(" => {
                let mut xs = Vec::new();
                while tokens.get(*at).copied() != Some(")") {
                    xs.push(Self::read_at(tokens, at, depth + 1, what, max_depth)?);
                }
                *at += 1;
                Ok(Expr::List(xs))
            }
            word => Ok(Expr::Word(word.to_owned())),
        }
    }

    pub(crate) fn word(&self) -> Result<&str> {
        match self {
            Self::Word(s) => Ok(s),
            Self::List(_) => Err(error("expected an identifier or literal")),
        }
    }

    pub(crate) fn list(&self) -> Result<&[Expr]> {
        match self {
            Self::List(xs) => Ok(xs),
            Self::Word(_) => Err(error("expected a parenthesized form")),
        }
    }

    /// The body of `(tag ...)`.
    pub(crate) fn tagged(&self, tag: &str) -> Result<&[Expr]> {
        let xs = self.list()?;
        if xs.first().map(Self::word).transpose()? != Some(tag) {
            return Err(error(format!("expected ({tag} ...)")));
        }
        Ok(&xs[1..])
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Word(s) => f.write_str(s),
            Self::List(xs) => {
                f.write_str("(")?;
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    x.fmt(f)?;
                }
                f.write_str(")")
            }
        }
    }
}

fn atom(expr: &Expr) -> Result<Atom> {
    let mut xs = expr.list()?;
    let negated = xs.first().map(Expr::word).transpose()? == Some("not");
    if negated {
        if xs.len() != 2 {
            return Err(error("not takes exactly one atom"));
        }
        xs = xs[1].list()?;
    }
    let name = xs.first().ok_or_else(|| error("empty atom"))?.word()?;
    let args = xs[1..]
        .iter()
        .map(|x| {
            let s = x.word()?;
            Ok(Value::literal(s).map_or_else(|| Term::Role(s.to_owned()), Term::Literal))
        })
        .collect::<Result<_>>()?;
    Ok(Atom { predicate: name.to_owned(), args, negated })
}

/// The roles of `(roles (sort name) ...)`.
fn roles(expr: &Expr) -> Result<Vec<Role>> {
    expr.tagged("roles")?
        .iter()
        .map(|role| {
            let pair = role.list()?;
            if pair.len() != 2 {
                return Err(error("role syntax is (sort name)"));
            }
            let sort = match pair[0].word()? {
                "context" => Sort::Context,
                "chair" => Sort::Chair,
                "domino" => Sort::Domino,
                _ => return Err(error("a role's sort is context, chair, or domino")),
            };
            Ok(Role { name: pair[1].word()?.to_owned(), sort })
        })
        .collect()
}

/// The clauses of a pattern or of a `none` body.
fn body(clauses: &[Expr]) -> Result<Pattern> {
    let mut pattern = Pattern::default();
    for clause in clauses {
        let xs = clause.list()?;
        match xs.first().map(Expr::word).transpose()? {
            Some("same") => {
                if xs.len() < 3 {
                    return Err(error("same requires at least two role names"));
                }
                pattern.equal.push(xs[1..].iter().map(|x| x.word().map(str::to_owned)).collect::<Result<_>>()?);
            }
            Some("none") => {
                let roles = roles(xs.get(1).ok_or_else(|| error("none syntax is (none (roles ...) clause ...)"))?)?;
                pattern.nones.push(NoneClause { roles, body: body(&xs[2..])? });
            }
            _ => pattern.atoms.push(atom(clause)?),
        }
    }
    Ok(pattern)
}

impl Query {
    pub(crate) fn from_expr(tree: &Expr) -> Result<Query> {
        let forms = tree.tagged("query")?;
        if forms.len() < 2 {
            return Err(error("a query requires roles and out forms"));
        }
        let mut query = Query { roles: roles(&forms[0])?, ..Query::default() };
        query.outputs = forms[1].tagged("out")?.iter().map(|x| x.word().map(str::to_owned)).collect::<Result<_>>()?;
        for form in &forms[2..] {
            query.patterns.push(body(form.tagged("pattern")?)?);
        }
        Ok(query)
    }
}

impl FromStr for Query {
    type Err = Error;

    /// At most 1 MiB of text, nested at most 32 deep.
    fn from_str(source: &str) -> Result<Query> {
        Query::from_expr(&Expr::read(source, "query", 1 << 20, 32)?)
    }
}

fn write_roles(f: &mut fmt::Formatter<'_>, roles: &[Role]) -> fmt::Result {
    write!(f, "(roles")?;
    for role in roles {
        write!(f, " ({} {})", role.sort, role.name)?;
    }
    write!(f, ")")
}

/// A body's clauses, one per line at `indent`: equalities, atoms, then nones.
fn write_body(f: &mut fmt::Formatter<'_>, pattern: &Pattern, indent: usize) -> fmt::Result {
    let pad = " ".repeat(indent);
    for group in &pattern.equal {
        write!(f, "\n{pad}(same {})", group.join(" "))?;
    }
    for atom in &pattern.atoms {
        let (open, close) = if atom.negated { ("(not ", ")") } else { ("", "") };
        write!(f, "\n{pad}{open}({}", atom.predicate)?;
        for arg in &atom.args {
            match arg {
                Term::Role(s) => write!(f, " {s}")?,
                Term::Literal(v) => write!(f, " {v}")?,
            }
        }
        write!(f, "){close}")?;
    }
    for none in &pattern.nones {
        write!(f, "\n{pad}(none ")?;
        write_roles(f, &none.roles)?;
        write_body(f, &none.body, indent + 2)?;
        write!(f, ")")?;
    }
    Ok(())
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(query\n  ")?;
        write_roles(f, &self.roles)?;
        write!(f, "\n  (out")?;
        for name in &self.outputs {
            write!(f, " {name}")?;
        }
        write!(f, ")")?;
        for pattern in &self.patterns {
            write!(f, "\n  (pattern")?;
            write_body(f, pattern, 4)?;
            write!(f, ")")?;
        }
        write!(f, ")")
    }
}
