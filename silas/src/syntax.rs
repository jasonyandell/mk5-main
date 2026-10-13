//! The surface language: sorts and values, the reader (with source spans and
//! comments), the `query` form, named definitions (`define`, `include`), and
//! the canonical formatter.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

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

    /// Every value a role of this sort ranges over; none for a literal-only sort.
    pub(crate) fn values(self) -> &'static [Value] {
        match self {
            Self::Context => &CONTEXT_VALUES,
            Self::Chair => &CHAIR_VALUES,
            Self::Domino => &DOMINO_VALUES,
            Self::Number | Self::Team => &[],
        }
    }
}

const CONTEXT_VALUES: [Value; Context::COUNT] = {
    let mut out = [Value::Context(Context::ALL[0]); Context::COUNT];
    let mut i = 0;
    while i < Context::COUNT {
        out[i] = Value::Context(Context::ALL[i]);
        i += 1;
    }
    out
};

const CHAIR_VALUES: [Value; Seat::COUNT] =
    [Value::Chair(Seat::ALL[0]), Value::Chair(Seat::ALL[1]), Value::Chair(Seat::ALL[2]), Value::Chair(Seat::ALL[3])];

const DOMINO_VALUES: [Value; Domino::COUNT] = {
    let mut out = [Value::Domino(Domino::ALL[0]); Domino::COUNT];
    let mut i = 0;
    while i < Domino::COUNT {
        out[i] = Value::Domino(Domino::ALL[i]);
        i += 1;
    }
    out
};

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

// ----------------------------------------------------------------- locations

/// A text a span points into: its name (a file path, or what was parsed) and
/// its contents.
#[derive(Debug)]
pub(crate) struct Source {
    name: String,
    text: String,
}

/// Where a piece of syntax came from. A location is not content: every two
/// spans are equal, so parsed and built syntax compare by what they say. A
/// span made in Rust (`Span::default()`) points nowhere.
#[derive(Clone, Default)]
pub struct Span {
    start: usize,
    end: usize,
    source: Option<Arc<Source>>,
}

impl PartialEq for Span {
    fn eq(&self, _: &Span) -> bool {
        true
    }
}

impl Eq for Span {}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line_column() {
            Some((line, column)) => write!(f, "{line}:{column}"),
            None => f.write_str("-"),
        }
    }
}

impl Span {
    /// The one-based line and column of the span's start, when it has a
    /// source.
    pub fn line_column(&self) -> Option<(usize, usize)> {
        let text = &self.source.as_ref()?.text;
        let before = &text[..self.start];
        let line = before.matches('\n').count() + 1;
        let column = before.rfind('\n').map_or(before.len(), |i| before.len() - i - 1) + 1;
        Some((line, column))
    }

    /// An error at this span: the message, and when the span has a source,
    /// its name, line and column, the line itself and a caret under the span.
    pub(crate) fn error(&self, message: impl Into<String>) -> Error {
        let message = message.into();
        let (Some(source), Some((line, column))) = (&self.source, self.line_column()) else {
            return error(message);
        };
        let start = self.start - (column - 1);
        let end = source.text[start..].find('\n').map_or(source.text.len(), |i| start + i);
        let shown = &source.text[start..end];
        let width = self.end.min(end).saturating_sub(self.start).max(1);
        let gutter = " ".repeat(line.to_string().len());
        error(format!(
            "{}:{line}:{column}: {message}\n{gutter} |\n{line} | {shown}\n{gutter} | {}{}",
            source.name,
            " ".repeat(column - 1),
            "^".repeat(width)
        ))
    }
}

// --------------------------------------------------------------- the reader

/// A parsed s-expression, with its span. Comments are kept only when the
/// reader is asked for them (the formatter); everything else reads words and
/// lists.
#[derive(Clone, Debug)]
pub(crate) struct Expr {
    pub(crate) kind: Kind,
    pub(crate) span: Span,
    /// The source has an empty line just before this expression.
    pub(crate) blank_before: bool,
}

#[derive(Clone, Debug)]
pub(crate) enum Kind {
    Word(String),
    List(Vec<Expr>),
    Comment(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Open,
    Close,
    Word(String),
    Comment(String),
}

struct Lexed {
    token: Token,
    start: usize,
    end: usize,
    blank_before: bool,
}

fn lex(text: &str) -> Vec<Lexed> {
    let mut out = Vec::new();
    let mut chars = text.char_indices().peekable();
    let mut newlines = 0;
    while let Some((i, c)) = chars.next() {
        let blank_before = newlines >= 2;
        let token = match c {
            '\n' => {
                newlines += 1;
                continue;
            }
            c if c.is_whitespace() => continue,
            '(' => (Token::Open, i + 1),
            ')' => (Token::Close, i + 1),
            ';' => {
                let mut end = text.len();
                while let Some((j, d)) = chars.peek() {
                    if *d == '\n' {
                        end = *j;
                        break;
                    }
                    chars.next();
                }
                (Token::Comment(text[i..end].trim_end().to_owned()), end)
            }
            _ => {
                let mut end = text.len();
                while let Some((j, d)) = chars.peek() {
                    if d.is_whitespace() || "();".contains(*d) {
                        end = *j;
                        break;
                    }
                    chars.next();
                }
                (Token::Word(text[i..end].to_owned()), end)
            }
        };
        newlines = 0;
        out.push(Lexed { token: token.0, start: i, end: token.1, blank_before });
    }
    out
}

impl Expr {
    /// Read every top-level form of `text`, named `name` in errors, at most
    /// `max_bytes` long and `max_depth` deep. Comments are kept when asked.
    pub(crate) fn read_all(
        text: &str,
        name: &str,
        max_bytes: usize,
        max_depth: usize,
        comments: bool,
    ) -> Result<Vec<Expr>> {
        let source = Arc::new(Source { name: name.to_owned(), text: text.to_owned() });
        let span = |start, end| Span { start, end, source: Some(source.clone()) };
        if text.len() > max_bytes {
            return Err(span(0, 0).error(format!("{name} exceeds {max_bytes} bytes")));
        }
        let tokens: Vec<Lexed> =
            lex(text).into_iter().filter(|t| comments || !matches!(t.token, Token::Comment(_))).collect();
        // A stack of open lists: each its opening token and its items so far.
        let mut stack: Vec<(usize, Vec<Expr>)> = vec![(0, Vec::new())];
        for (k, t) in tokens.iter().enumerate() {
            match &t.token {
                Token::Open => {
                    if stack.len() > max_depth {
                        return Err(span(t.start, t.end).error(format!("nesting exceeds {max_depth}")));
                    }
                    stack.push((k, Vec::new()));
                }
                Token::Close => {
                    if stack.len() == 1 {
                        return Err(span(t.start, t.end).error("unexpected closing parenthesis"));
                    }
                    let (open, items) = stack.pop().expect("an open list");
                    let o = &tokens[open];
                    let expr =
                        Expr { kind: Kind::List(items), span: span(o.start, t.end), blank_before: o.blank_before };
                    stack.last_mut().expect("an enclosing list").1.push(expr);
                }
                Token::Word(w) | Token::Comment(w) => {
                    let kind = match t.token {
                        Token::Comment(_) => Kind::Comment(w.clone()),
                        _ => Kind::Word(w.clone()),
                    };
                    let expr = Expr { kind, span: span(t.start, t.end), blank_before: t.blank_before };
                    stack.last_mut().expect("an enclosing list").1.push(expr);
                }
            }
        }
        if stack.len() > 1 {
            let o = &tokens[stack.last().expect("an open list").0];
            return Err(span(o.start, o.end).error("this parenthesis is never closed"));
        }
        Ok(stack.pop().expect("the top level").1)
    }

    pub(crate) fn word(&self) -> Result<&str> {
        match &self.kind {
            Kind::Word(s) => Ok(s),
            _ => Err(self.span.error("expected a name or a literal")),
        }
    }

    pub(crate) fn list(&self) -> Result<&[Expr]> {
        match &self.kind {
            Kind::List(xs) => Ok(xs),
            _ => Err(self.span.error("expected a parenthesized form")),
        }
    }

    /// The first word of a list.
    pub(crate) fn head(&self) -> Option<&str> {
        match &self.kind {
            Kind::List(xs) => match xs.first().map(|x| &x.kind) {
                Some(Kind::Word(w)) => Some(w),
                _ => None,
            },
            _ => None,
        }
    }

    /// The body of `(tag ...)`.
    pub(crate) fn tagged(&self, tag: &str) -> Result<&[Expr]> {
        if self.head() != Some(tag) {
            return Err(self.span.error(format!("expected ({tag} ...)")));
        }
        Ok(&self.list()?[1..])
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            Kind::Word(s) | Kind::Comment(s) => f.write_str(s),
            Kind::List(xs) => {
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

// ----------------------------------------------------------------- queries

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Role {
    pub name: String,
    pub sort: Sort,
    pub span: Span,
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
    pub span: Span,
}

/// One conjunctive branch of a query, or the body of a scope.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    /// Same-sort equality groups over the roles declared with this body,
    /// transitively closed when compiled. Roles of one sort declared together
    /// and in different groups denote DISTINCT values.
    pub equal: Vec<Vec<String>>,
    pub atoms: Vec<Atom>,
    pub scopes: Vec<Scope>,
    pub span: Span,
}

/// How a scope binds its roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantifier {
    /// `(some (roles ...) clause ...)`: some binding satisfies the body.
    Some,
    /// `(none (roles ...) clause ...)`: no binding satisfies the body.
    None,
}

/// `(some ...)` or `(none ...)`: a stratified scope over its own roles, read
/// once the enclosing roles it names are bound, binding nothing outside
/// itself. Three-valued: `some` is Kleene's existential, `none` its negation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub quantifier: Quantifier,
    pub roles: Vec<Role>,
    pub body: Pattern,
    pub span: Span,
}

/// A relation: the set of output tuples for which some pattern has a
/// satisfying assignment of the remaining (existential) roles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Query {
    pub roles: Vec<Role>,
    pub outputs: Vec<String>,
    pub patterns: Vec<Pattern>,
    pub span: Span,
}

pub(crate) fn valid_name(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Words the grammar reserves: no predicate or definition takes them.
pub(crate) const RESERVED: [&str; 9] = ["query", "roles", "out", "pattern", "same", "not", "none", "some", "define"];

/// Every name visible at a point of a body, with its sort.
type Names = BTreeMap<String, Sort>;

/// The roles of `(roles (sort name) ...)`.
fn roles(expr: &Expr) -> Result<Vec<Role>> {
    sorted(expr, false)
}

/// The names of `(roles (sort name) ...)`; a definition's parameters may also
/// be numbers and teams, which its callers pass as literals.
fn sorted(expr: &Expr, literal_sorts: bool) -> Result<Vec<Role>> {
    expr.tagged("roles")?
        .iter()
        .map(|role| {
            let pair = role.list()?;
            if pair.len() != 2 {
                return Err(role.span.error("role syntax is (sort name)"));
            }
            let sort = match pair[0].word()? {
                "context" => Sort::Context,
                "chair" => Sort::Chair,
                "domino" => Sort::Domino,
                "number" if literal_sorts => Sort::Number,
                "team" if literal_sorts => Sort::Team,
                _ if literal_sorts => {
                    return Err(pair[0].span.error("a parameter's sort is context, chair, domino, number, or team"));
                }
                _ => return Err(pair[0].span.error("a role's sort is context, chair, or domino")),
            };
            Ok(Role { name: pair[1].word()?.to_owned(), sort, span: role.span.clone() })
        })
        .collect()
}

/// A named definition: its parameters, in order, and its body (one
/// conjunction, its own definitions already expanded).
#[derive(Clone, Debug)]
struct Define {
    params: Vec<Role>,
    body: Pattern,
}

/// The built-in libraries `(include name)` may name.
const LIBRARIES: [(&str, &str); 1] = [("book", include_str!("../lib/book.silas"))];

/// The definitions in force while a text is read.
#[derive(Clone, Debug, Default)]
pub(crate) struct Library {
    defines: BTreeMap<String, Define>,
    included: BTreeSet<String>,
}

impl Library {
    /// Read a whole program: its `include` and `define` forms, then exactly
    /// one final form tagged `what`. Returns the library and the final form.
    pub(crate) fn program(
        text: &str,
        name: &str,
        what: &str,
        max_bytes: usize,
        max_depth: usize,
    ) -> Result<(Library, Expr)> {
        let forms = Expr::read_all(text, name, max_bytes, max_depth, false)?;
        let Some((last, rest)) = forms.split_last() else {
            return Err(error(format!("{name}: expected a ({what} ...) form, found nothing")));
        };
        let mut library = Library::default();
        for form in rest {
            library.form(form)?;
        }
        if last.head() != Some(what) {
            return Err(last.span.error(format!("expected a final ({what} ...) form")));
        }
        Ok((library, last.clone()))
    }

    /// One `(include name)` or `(define ...)`.
    fn form(&mut self, form: &Expr) -> Result<()> {
        match form.head() {
            Some("include") => match form.tagged("include")? {
                [name] => self.include(name),
                _ => Err(form.span.error("include syntax is (include name)")),
            },
            Some("define") => self.define(form),
            _ => Err(form.span.error("before the final form, only (include ...) and (define ...) may appear")),
        }
    }

    fn include(&mut self, name: &Expr) -> Result<()> {
        let word = name.word()?;
        let Some((_, text)) = LIBRARIES.iter().find(|(n, _)| *n == word) else {
            let known: Vec<&str> = LIBRARIES.iter().map(|(n, _)| *n).collect();
            return Err(name.span.error(format!("no library {word}; the libraries are {}", known.join(", "))));
        };
        if !self.included.insert(word.to_owned()) {
            return Ok(());
        }
        for form in Expr::read_all(text, &format!("library {word}"), 1 << 20, 32, false)? {
            self.form(&form)?;
        }
        Ok(())
    }

    /// `(define name (roles (sort param) ...) clause ...)`.
    fn define(&mut self, form: &Expr) -> Result<()> {
        let body = form.tagged("define")?;
        let [name, params, clauses @ ..] = body else {
            return Err(form.span.error("define syntax is (define name (roles (sort param) ...) clause ...)"));
        };
        let word = name.word()?;
        let standard = crate::Registry::standard();
        if !valid_name(word)
            || Value::literal(word).is_some()
            || RESERVED.contains(&word)
            || standard.predicates.contains_key(word)
        {
            return Err(name
                .span
                .error(format!("{word} cannot name a definition: it is reserved, a literal, or a predicate")));
        }
        if self.defines.contains_key(word) {
            return Err(name.span.error(format!("{word} is already defined")));
        }
        let params = sorted(params, true)?;
        let mut names = Names::new();
        for p in &params {
            if !valid_name(&p.name)
                || Value::literal(&p.name).is_some()
                || names.insert(p.name.clone(), p.sort).is_some()
            {
                return Err(p.span.error(format!("invalid or duplicate parameter {}", p.name)));
            }
        }
        let body = self.body(clauses, &names, form.span.clone())?;
        if !body.equal.is_empty() {
            return Err(form
                .span
                .error("a definition's body has no roles of its own to join with same; use (some ...)"));
        }
        self.defines.insert(word.to_owned(), Define { params, body });
        Ok(())
    }

    /// The clauses of a pattern or a scope body, with every definition they
    /// call expanded.
    fn body(&self, clauses: &[Expr], names: &Names, span: Span) -> Result<Pattern> {
        let mut pattern = Pattern { span, ..Pattern::default() };
        for clause in clauses {
            let xs = clause.list()?;
            match clause.head() {
                Some("same") => {
                    if xs.len() < 3 {
                        return Err(clause.span.error("same requires at least two role names"));
                    }
                    pattern.equal.push(xs[1..].iter().map(|x| x.word().map(str::to_owned)).collect::<Result<_>>()?);
                }
                Some(quantifier @ ("none" | "some")) => {
                    let declared = xs.get(1).ok_or_else(|| {
                        clause.span.error(format!("{quantifier} syntax is ({quantifier} (roles ...) clause ...)"))
                    })?;
                    let roles = roles(declared)?;
                    let mut inner = names.clone();
                    inner.extend(roles.iter().map(|r| (r.name.clone(), r.sort)));
                    let body = self.body(&xs[2..], &inner, clause.span.clone())?;
                    let quantifier = if quantifier == "none" { Quantifier::None } else { Quantifier::Some };
                    pattern.scopes.push(Scope { quantifier, roles, body, span: clause.span.clone() });
                }
                _ => {
                    let atom = atom(clause)?;
                    match self.defines.get(&atom.predicate) {
                        Some(define) => self.expand(define, atom, names, &mut pattern)?,
                        None => pattern.atoms.push(atom),
                    }
                }
            }
        }
        Ok(pattern)
    }

    /// A call of a definition, expanded in place: its body with the
    /// arguments for its parameters and its own roles renamed apart from the
    /// names around the call. Negated, it becomes a `none` (or a negated atom,
    /// when its body is one atom).
    fn expand(&self, define: &Define, call: Atom, names: &Names, pattern: &mut Pattern) -> Result<()> {
        if call.args.len() != define.params.len() {
            return Err(call.span.error(format!(
                "{} takes {} argument(s), got {}",
                call.predicate,
                define.params.len(),
                call.args.len()
            )));
        }
        let mut substitution = BTreeMap::new();
        let mut taken: BTreeSet<String> = names.keys().cloned().collect();
        for (param, arg) in define.params.iter().zip(&call.args) {
            let sort = match arg {
                Term::Literal(v) => Some(v.sort()),
                Term::Role(name) => {
                    taken.insert(name.clone());
                    names.get(name).copied()
                }
            };
            if sort.is_some_and(|s| s != param.sort) {
                return Err(call.span.error(format!(
                    "{}'s parameter {} is a {}, got a {}",
                    call.predicate,
                    param.name,
                    param.sort,
                    sort.expect("a sort")
                )));
            }
            substitution.insert(param.name.clone(), arg.clone());
        }
        let body = instantiate(&define.body, &substitution, &mut taken, &call.span);
        if !call.negated {
            pattern.atoms.extend(body.atoms);
            pattern.scopes.extend(body.scopes);
        } else if let ([one], []) = (body.atoms.as_slice(), body.scopes.as_slice()) {
            pattern.atoms.push(Atom { negated: !one.negated, ..one.clone() });
        } else if let ([], [one]) = (body.atoms.as_slice(), body.scopes.as_slice())
            && one.quantifier == Quantifier::Some
        {
            pattern.scopes.push(Scope { quantifier: Quantifier::None, ..one.clone() });
        } else {
            pattern.scopes.push(Scope { quantifier: Quantifier::None, roles: vec![], body, span: call.span });
        }
        Ok(())
    }
}

/// A fresh name for `name`, apart from every name in `taken`.
fn fresh(name: &str, taken: &mut BTreeSet<String>) -> String {
    let chosen = (1..)
        .map(|k| if k == 1 { name.to_owned() } else { format!("{name}-{k}") })
        .find(|n| !taken.contains(n))
        .expect("an unused name");
    taken.insert(chosen.clone());
    chosen
}

/// `pattern` with each name substituted (parameters by arguments, bound
/// roles by fresh names) and every span moved to the call.
fn instantiate(
    pattern: &Pattern,
    substitution: &BTreeMap<String, Term>,
    taken: &mut BTreeSet<String>,
    at: &Span,
) -> Pattern {
    let rename = |name: &String| match substitution.get(name) {
        Some(Term::Role(n)) => n.clone(),
        _ => name.clone(),
    };
    let atoms = pattern
        .atoms
        .iter()
        .map(|a| Atom {
            predicate: a.predicate.clone(),
            args: a
                .args
                .iter()
                .map(|t| match t {
                    Term::Role(n) => substitution.get(n).cloned().unwrap_or_else(|| t.clone()),
                    Term::Literal(_) => t.clone(),
                })
                .collect(),
            negated: a.negated,
            span: at.clone(),
        })
        .collect();
    let scopes = pattern
        .scopes
        .iter()
        .map(|s| {
            let mut inner = substitution.clone();
            let roles = s
                .roles
                .iter()
                .map(|r| {
                    let name = fresh(&r.name, taken);
                    inner.insert(r.name.clone(), Term::Role(name.clone()));
                    Role { name, sort: r.sort, span: at.clone() }
                })
                .collect();
            Scope { quantifier: s.quantifier, roles, body: instantiate(&s.body, &inner, taken, at), span: at.clone() }
        })
        .collect();
    Pattern {
        equal: pattern.equal.iter().map(|g| g.iter().map(rename).collect()).collect(),
        atoms,
        scopes,
        span: at.clone(),
    }
}

fn atom(expr: &Expr) -> Result<Atom> {
    let mut xs = expr.list()?;
    let negated = expr.head() == Some("not");
    if negated {
        if xs.len() != 2 {
            return Err(expr.span.error("not takes exactly one atom"));
        }
        xs = xs[1].list()?;
    }
    let name = xs.first().ok_or_else(|| expr.span.error("an empty atom"))?.word()?;
    let args = xs[1..]
        .iter()
        .map(|x| {
            let s = x.word()?;
            Ok(Value::literal(s).map_or_else(|| Term::Role(s.to_owned()), Term::Literal))
        })
        .collect::<Result<_>>()?;
    Ok(Atom { predicate: name.to_owned(), args, negated, span: expr.span.clone() })
}

impl Query {
    pub(crate) fn from_expr(library: &Library, tree: &Expr) -> Result<Query> {
        let forms = tree.tagged("query")?;
        if forms.len() < 2 {
            return Err(tree.span.error("a query requires (roles ...) and (out ...) forms"));
        }
        let mut query = Query { roles: roles(&forms[0])?, span: tree.span.clone(), ..Query::default() };
        query.outputs = forms[1].tagged("out")?.iter().map(|x| x.word().map(str::to_owned)).collect::<Result<_>>()?;
        let names: Names = query.roles.iter().map(|r| (r.name.clone(), r.sort)).collect();
        for form in &forms[2..] {
            query.patterns.push(library.body(form.tagged("pattern")?, &names, form.span.clone())?);
        }
        Ok(query)
    }

    /// Parse a query text, naming it `name` in errors: any `include` and
    /// `define` forms, then the `(query ...)`. At most 1 MiB, nested at most 32
    /// deep.
    pub fn parse(text: &str, name: &str) -> Result<Query> {
        let (library, form) = Library::program(text, name, "query", 1 << 20, 32)?;
        Query::from_expr(&library, &form)
    }
}

impl FromStr for Query {
    type Err = Error;

    fn from_str(source: &str) -> Result<Query> {
        Query::parse(source, "query")
    }
}

fn write_roles(f: &mut fmt::Formatter<'_>, roles: &[Role]) -> fmt::Result {
    write!(f, "(roles")?;
    for role in roles {
        write!(f, " ({} {})", role.sort, role.name)?;
    }
    write!(f, ")")
}

/// A body's clauses, one per line at `indent`: equalities, atoms, then scopes.
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
    for scope in &pattern.scopes {
        let tag = match scope.quantifier {
            Quantifier::Some => "some",
            Quantifier::None => "none",
        };
        write!(f, "\n{pad}({tag} ")?;
        write_roles(f, &scope.roles)?;
        write_body(f, &scope.body, indent + 2)?;
        write!(f, ")")?;
    }
    Ok(())
}

impl fmt::Display for Query {
    /// The query with every definition expanded: it reads back as itself.
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

// --------------------------------------------------------------- formatting

/// The widest line the formatter writes, when a form allows it.
const WIDTH: usize = 100;

/// Heads whose forms always take a line per item.
const BROKEN: [&str; 6] = ["policy", "rule", "query", "pattern", "define", "dominates"];

/// Format Silas text canonically: comments kept, each on its own line;
/// blank lines kept where the source has one (at most one); a form on one
/// line when it fits in 100 columns and holds no comment or blank line, and
/// is not a policy, rule, query, pattern, definition or dominance; otherwise its leading
/// words (and a following `(roles ...)`) on its first line and each other
/// item on a line of its own, two columns in. Formatting is idempotent.
pub fn format(text: &str, name: &str) -> Result<String> {
    let forms = Expr::read_all(text, name, 4 << 20, 64, true)?;
    let mut out = String::new();
    for (i, form) in forms.iter().enumerate() {
        if i > 0 {
            out.push('\n');
            if form.blank_before {
                out.push('\n');
            }
        }
        write_form(form, 0, &mut out);
    }
    out.push('\n');
    Ok(out)
}

/// The form on one line, when it may be written so.
fn flat(expr: &Expr) -> Option<String> {
    match &expr.kind {
        Kind::Word(w) => Some(w.clone()),
        Kind::Comment(_) => None,
        Kind::List(xs) => {
            if expr.head().is_some_and(|h| BROKEN.contains(&h)) || xs.iter().skip(1).any(|x| x.blank_before) {
                return None;
            }
            let items: Option<Vec<String>> = xs.iter().map(flat).collect();
            Some(format!("({})", items?.join(" ")))
        }
    }
}

fn column(out: &str) -> usize {
    out.len() - out.rfind('\n').map_or(0, |i| i + 1)
}

fn write_form(expr: &Expr, indent: usize, out: &mut String) {
    if let Some(line) = flat(expr).filter(|line| column(out) + line.len() <= WIDTH) {
        out.push_str(&line);
        return;
    }
    let Kind::List(xs) = &expr.kind else {
        if let Kind::Word(w) | Kind::Comment(w) = &expr.kind {
            out.push_str(w);
        }
        return;
    };
    out.push('(');
    let mut rest = xs.as_slice();
    let mut first = true;
    while let Some((x, tail)) = rest.split_first() {
        let Kind::Word(w) = &x.kind else { break };
        if !first {
            out.push(' ');
        }
        out.push_str(w);
        first = false;
        rest = tail;
    }
    if !first
        && let Some((x, tail)) = rest.split_first()
        && x.head() == Some("roles")
        && let Some(line) = flat(x).filter(|line| column(out) + 1 + line.len() <= WIDTH)
    {
        out.push(' ');
        out.push_str(&line);
        rest = tail;
    }
    let pad = " ".repeat(indent + 2);
    for x in rest {
        out.push('\n');
        if x.blank_before {
            out.push('\n');
        }
        out.push_str(&pad);
        write_form(x, indent + 2, out);
    }
    if rest.last().is_some_and(|x| matches!(x.kind, Kind::Comment(_))) {
        out.push('\n');
        out.push_str(&" ".repeat(indent));
    }
    out.push(')');
}
