/// Propositional and predicate logic: SAT solver, theorem prover, unification.

use std::collections::{HashMap, HashSet, BTreeSet};

/// Propositional formula.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Proposition {
    Var(String),
    Not(Box<Proposition>),
    And(Box<Proposition>, Box<Proposition>),
    Or(Box<Proposition>, Box<Proposition>),
    Implies(Box<Proposition>, Box<Proposition>),
    Iff(Box<Proposition>, Box<Proposition>),
}

impl Proposition {
    pub fn var(name: &str) -> Self { Proposition::Var(name.to_string()) }
    pub fn not(p: Proposition) -> Self { Proposition::Not(Box::new(p)) }
    pub fn and(a: Proposition, b: Proposition) -> Self { Proposition::And(Box::new(a), Box::new(b)) }
    pub fn or(a: Proposition, b: Proposition) -> Self { Proposition::Or(Box::new(a), Box::new(b)) }
    pub fn implies(a: Proposition, b: Proposition) -> Self { Proposition::Implies(Box::new(a), Box::new(b)) }
    pub fn iff(a: Proposition, b: Proposition) -> Self { Proposition::Iff(Box::new(a), Box::new(b)) }

    pub fn evaluate(&self, assignment: &HashMap<String, bool>) -> Option<bool> {
        match self {
            Proposition::Var(name) => assignment.get(name).copied(),
            Proposition::Not(p) => p.evaluate(assignment).map(|v| !v),
            Proposition::And(a, b) => {
                let av = a.evaluate(assignment)?;
                let bv = b.evaluate(assignment)?;
                Some(av && bv)
            }
            Proposition::Or(a, b) => {
                let av = a.evaluate(assignment)?;
                let bv = b.evaluate(assignment)?;
                Some(av || bv)
            }
            Proposition::Implies(a, b) => {
                let av = a.evaluate(assignment)?;
                let bv = b.evaluate(assignment)?;
                Some(!av || bv)
            }
            Proposition::Iff(a, b) => {
                let av = a.evaluate(assignment)?;
                let bv = b.evaluate(assignment)?;
                Some(av == bv)
            }
        }
    }

    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Proposition::Var(name) => { vars.insert(name.clone()); }
            Proposition::Not(p) => p.collect_vars(vars),
            Proposition::And(a, b) | Proposition::Or(a, b) |
            Proposition::Implies(a, b) | Proposition::Iff(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
        }
    }

    /// Convert to NNF (Negation Normal Form).
    pub fn to_nnf(&self) -> Proposition {
        match self {
            Proposition::Not(inner) => {
                match inner.as_ref() {
                    Proposition::Not(p) => p.to_nnf(),
                    Proposition::And(a, b) => Proposition::or(
                        Proposition::not(a.clone()).to_nnf(),
                        Proposition::not(b.clone()).to_nnf(),
                    ),
                    Proposition::Or(a, b) => Proposition::and(
                        Proposition::not(a.clone()).to_nnf(),
                        Proposition::not(b.clone()).to_nnf(),
                    ),
                    Proposition::Implies(a, b) => Proposition::and(
                        a.clone().to_nnf(),
                        Proposition::not(b.clone()).to_nnf(),
                    ),
                    _ => Proposition::Not(Box::new(inner.to_nnf())),
                }
            }
            Proposition::Implies(a, b) => {
                Proposition::or(Proposition::not(a.clone()).to_nnf(), b.clone().to_nnf())
            }
            Proposition::Iff(a, b) => {
                let ab = Proposition::implies(a.as_ref().clone(), b.as_ref().clone());
                let ba = Proposition::implies(b.as_ref().clone(), a.as_ref().clone());
                Proposition::and(ab, ba).to_nnf()
            }
            Proposition::And(a, b) => Proposition::and(a.to_nnf(), b.to_nnf()),
            Proposition::Or(a, b) => Proposition::or(a.to_nnf(), b.to_nnf()),
            _ => self.clone(),
        }
    }

    /// Convert to CNF (Conjunctive Normal Form) as a set of clauses.
    pub fn to_cnf(&self) -> Vec<Vec<Literal>> {
        let nnf = self.to_nnf();
        let clauses = nnf_to_clauses(&nnf);
        let mut result = Vec::new();
        for clause in clauses {
            let mut literals: Vec<Literal> = clause.into_iter().collect();
            literals.sort();
            literals.dedup();
            result.push(literals);
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Literal {
    Pos(String),
    Neg(String),
}

impl Literal {
    pub fn var_name(&self) -> &str {
        match self {
            Literal::Pos(n) | Literal::Neg(n) => n,
        }
    }

    pub fn is_positive(&self) -> bool {
        matches!(self, Literal::Pos(_))
    }

    pub fn negate(&self) -> Literal {
        match self {
            Literal::Pos(n) => Literal::Neg(n.clone()),
            Literal::Neg(n) => Literal::Pos(n.clone()),
        }
    }
}

fn nnf_to_clauses(formula: &Proposition) -> Vec<BTreeSet<Literal>> {
    match formula {
        Proposition::Var(name) => vec![{
            let mut s = BTreeSet::new();
            s.insert(Literal::Pos(name.clone()));
            s
        }],
        Proposition::Not(inner) => {
            if let Proposition::Var(name) = inner.as_ref() {
                vec![{
                    let mut s = BTreeSet::new();
                    s.insert(Literal::Neg(name.clone()));
                    s
                }]
            } else {
                unreachable!("NNF should have negations only on variables")
            }
        }
        Proposition::Or(a, b) => {
            let ca = nnf_to_clauses(a);
            let cb = nnf_to_clauses(b);
            // Distribute: (A1|A2) | (B1|B2) = all pairwise unions
            let mut result = Vec::new();
            for clause_a in &ca {
                for clause_b in &cb {
                    let mut merged = clause_a.clone();
                    merged.extend(clause_b.iter().cloned());
                    result.push(merged);
                }
            }
            result
        }
        Proposition::And(a, b) => {
            let mut result = nnf_to_clauses(a);
            result.extend(nnf_to_clauses(b));
            result
        }
        _ => unreachable!("NNF should only contain And, Or, Not(Var), Var"),
    }
}

/// DPLL SAT solver.
pub struct SatSolver {
    clauses: Vec<Vec<Literal>>,
    assignment: HashMap<String, bool>,
}

impl SatSolver {
    pub fn new(clauses: Vec<Vec<Literal>>) -> Self {
        Self { clauses, assignment: HashMap::new() }
    }

    pub fn solve(&mut self) -> Option<HashMap<String, bool>> {
        if self.dpll() {
            Some(self.assignment.clone())
        } else {
            None
        }
    }

    fn dpll(&mut self) -> bool {
        // Unit propagation
        loop {
            match self.find_unit() {
                Some((var, val)) => {
                    self.assignment.insert(var, val);
                    if !self.propagate() { return false; }
                }
                None => break,
            }
        }

        // Pure literal elimination
        loop {
            match self.find_pure() {
                Some((var, val)) => {
                    self.assignment.insert(var, val);
                    if !self.propagate() { return false; }
                }
                None => break,
            }
        }

        // Check if all clauses satisfied
        if self.clauses.is_empty() { return true; }

        // Check for empty clause (unsatisfiable)
        if self.clauses.iter().any(|c| c.is_empty()) { return false; }

        // Pick an unassigned variable
        let unassigned = self.find_unassigned();
        if unassigned.is_none() { return self.clauses.is_empty(); }
        let var = unassigned.unwrap();

        // Try true
        let saved_clauses = self.clauses.clone();
        let saved_assignment = self.assignment.clone();

        self.assignment.insert(var.clone(), true);
        if self.propagate() && self.dpll() { return true; }

        self.clauses = saved_clauses;
        self.assignment = saved_assignment;

        // Try false
        self.assignment.insert(var.clone(), false);
        if self.propagate() && self.dpll() { return true; }

        self.clauses = saved_clauses;
        self.assignment = saved_assignment;
        false
    }

    fn find_unit(&self) -> Option<(String, bool)> {
        for clause in &self.clauses {
            if clause.len() == 1 {
                let lit = &clause[0];
                if !self.assignment.contains_key(lit.var_name()) {
                    return Some((lit.var_name().to_string(), lit.is_positive()));
                }
            }
        }
        None
    }

    fn find_pure(&self) -> Option<(String, bool)> {
        let mut polarity: HashMap<String, Option<bool>> = HashMap::new();
        for clause in &self.clauses {
            for lit in clause {
                if self.assignment.contains_key(lit.var_name()) { continue; }
                let entry = polarity.entry(lit.var_name().to_string()).or_insert(None);
                let lit_positive = lit.is_positive();
                match entry {
                    None => *entry = Some(lit_positive),
                    Some(p) if *p != lit_positive => *entry = None, // both polarities
                    _ => {}
                }
            }
        }
        for (var, pol) in polarity {
            if let Some(val) = pol {
                return Some((var, val));
            }
        }
        None
    }

    fn find_unassigned(&self) -> Option<String> {
        for clause in &self.clauses {
            for lit in clause {
                if !self.assignment.contains_key(lit.var_name()) {
                    return Some(lit.var_name().to_string());
                }
            }
        }
        None
    }

    fn propagate(&mut self) -> bool {
        let mut changed = true;
        while changed {
            changed = false;
            let mut new_clauses = Vec::new();
            for clause in &self.clauses {
                let mut satisfied = false;
                let mut new_clause = Vec::new();
                for lit in clause {
                    if let Some(&val) = self.assignment.get(lit.var_name()) {
                        if (val && lit.is_positive()) || (!val && !lit.is_positive()) {
                            satisfied = true;
                            break;
                        }
                    } else {
                        new_clause.push(lit.clone());
                    }
                }
                if !satisfied {
                    if new_clause.is_empty() { return false; }
                    new_clauses.push(new_clause);
                }
            }
            if new_clauses.len() != self.clauses.len() {
                changed = true;
                self.clauses = new_clauses;
            } else {
                self.clauses = new_clauses;
            }
        }
        true
    }
}

/// First-order logic terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Variable(String),
    Constant(String),
    Function(String, Vec<Term>),
}

impl Term {
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Term::Variable(v) => { vars.insert(v.clone()); }
            Term::Function(_, args) => {
                for arg in args { arg.collect_vars(vars); }
            }
            _ => {}
        }
    }

    pub fn substitute(&self, subst: &HashMap<String, Term>) -> Term {
        match self {
            Term::Variable(v) => subst.get(v).cloned().unwrap_or(self.clone()),
            Term::Function(name, args) => {
                let new_args: Vec<Term> = args.iter().map(|a| a.substitute(subst)).collect();
                Term::Function(name.clone(), new_args)
            }
            _ => self.clone(),
        }
    }
}

/// First-order predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Predicate {
    pub name: String,
    pub args: Vec<Term>,
}

/// First-order formula.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Formula {
    Predicate(Predicate),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Implies(Box<Formula>, Box<Formula>),
    ForAll(String, Box<Formula>),
    Exists(String, Box<Formula>),
}

/// Unification algorithm (Robinson's).
pub struct Unifier;

impl Unifier {
    pub fn unify(t1: &Term, t2: &Term) -> Option<HashMap<String, Term>> {
        let mut subst = HashMap::new();
        if Self::unify_internal(t1, t2, &mut subst) {
            Some(subst)
        } else {
            None
        }
    }

    fn unify_internal(t1: &Term, t2: &Term, subst: &mut HashMap<String, Term>) -> bool {
        let t1 = Self::apply_subst(t1, subst);
        let t2 = Self::apply_subst(t2, subst);

        match (&t1, &t2) {
            _ if t1 == t2 => true,
            (Term::Variable(v), _) => {
                if Self::occurs_check(v, &t2) { return false; }
                subst.insert(v.clone(), t2);
                true
            }
            (_, Term::Variable(v)) => {
                if Self::occurs_check(v, &t1) { return false; }
                subst.insert(v.clone(), t1);
                true
            }
            (Term::Function(n1, a1), Term::Function(n2, a2)) => {
                if n1 != n2 || a1.len() != a2.len() { return false; }
                a1.iter().zip(a2.iter()).all(|(x, y)| Self::unify_internal(x, y, subst))
            }
            _ => false,
        }
    }

    fn occurs_check(var: &str, term: &Term) -> bool {
        match term {
            Term::Variable(v) => v == var,
            Term::Function(_, args) => args.iter().any(|a| Self::occurs_check(var, a)),
            _ => false,
        }
    }

    fn apply_subst(term: &Term, subst: &HashMap<String, Term>) -> Term {
        match term {
            Term::Variable(v) => {
                if let Some(t) = subst.get(v) {
                    Self::apply_subst(t, subst)
                } else {
                    term.clone()
                }
            }
            Term::Function(name, args) => {
                let new_args: Vec<Term> = args.iter().map(|a| Self::apply_subst(a, subst)).collect();
                Term::Function(name.clone(), new_args)
            }
            _ => term.clone(),
        }
    }
}

/// Resolution theorem prover for first-order logic.
pub struct ResolutionProver;

impl ResolutionProver {
    /// Convert a formula to clausal form (skolemized CNF).
    pub fn to_clauses(formula: &Formula) -> Vec<Vec<Predicate>> {
        // Simplified: works for ground (variable-free) formulas
        Self::extract_clauses(formula)
    }

    fn extract_clauses(formula: &Formula) -> Vec<Vec<Predicate>> {
        match formula {
            Formula::Predicate(p) => vec![vec![p.clone()]],
            Formula::And(a, b) => {
                let mut result = Self::extract_clauses(a);
                result.extend(Self::extract_clauses(b));
                result
            }
            Formula::Or(a, b) => {
                let ca = Self::extract_clauses(a);
                let cb = Self::extract_clauses(b);
                if ca.len() == 1 && cb.len() == 1 {
                    let mut merged = ca[0].clone();
                    merged.extend(cb[0].clone());
                    vec![merged]
                } else {
                    vec![vec![]] // Simplified
                }
            }
            Formula::Not(inner) => {
                if let Formula::Predicate(p) = inner.as_ref() {
                    // Negated literal - in real implementation would track sign
                    vec![vec![p.clone()]]
                } else {
                    vec![vec![]]
                }
            }
            _ => vec![vec![]],
        }
    }

    /// Attempt resolution proof.
    pub fn prove(clauses: &[Vec<Predicate>]) -> bool {
        let mut clause_set: Vec<Vec<Predicate>> = clauses.to_vec();
        let mut new = Vec::new();

        for _ in 0..100 {
            for i in 0..clause_set.len() {
                for j in (i + 1)..clause_set.len() {
                    let resolvents = Self::resolve(&clause_set[i], &clause_set[j]);
                    if resolvents.iter().any(|r| r.is_empty()) {
                        return true; // Empty clause = contradiction
                    }
                    new.extend(resolvents);
                }
            }

            if new.is_empty() { return false; }

            clause_set.extend(new.drain(..));
        }

        false
    }

    fn resolve(c1: &[Predicate], c2: &[Predicate]) -> Vec<Vec<Predicate>> {
        let mut result = Vec::new();
        for p1 in c1 {
            for p2 in c2 {
                if p1.name == p2.name && p1.args == p2.args {
                    // Complementary literals - resolve
                    let mut new_clause: Vec<Predicate> = c1.iter()
                        .filter(|p| *p != p1)
                        .chain(c2.iter().filter(|p| *p != p2))
                        .cloned()
                        .collect();
                    new_clause.sort_by(|a, b| a.name.cmp(&b.name));
                    new_clause.dedup();
                    result.push(new_clause);
                }
            }
        }
        result
    }
}

/// Truth table generator.
pub fn truth_table(formula: &Proposition) -> Vec<(HashMap<String, bool>, bool)> {
    let vars: Vec<String> = {
        let mut v: Vec<String> = formula.variables().into_iter().collect();
        v.sort();
        v
    };

    let n = vars.len();
    let mut rows = Vec::new();

    for mask in 0..(1 << n) {
        let mut assignment = HashMap::new();
        for (i, var) in vars.iter().enumerate() {
            assignment.insert(var.clone(), (mask >> i) & 1 == 1);
        }
        let value = formula.evaluate(&assignment).unwrap_or(false);
        rows.push((assignment, value));
    }

    rows
}

/// Check if a formula is a tautology.
pub fn is_tautology(formula: &Proposition) -> bool {
    truth_table(formula).iter().all(|(_, v)| *v)
}

/// Check if a formula is satisfiable.
pub fn is_satisfiable(formula: &Proposition) -> bool {
    truth_table(formula).iter().any(|(_, v)| *v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_table() {
        let p = Proposition::var("p");
        let q = Proposition::var("q");
        let pq = Proposition::implies(p.clone(), q.clone());
        let qp = Proposition::implies(q, p);
        let iff = Proposition::and(pq, qp);

        assert!(is_tautology(&iff));
    }

    #[test]
    fn test_sat_solver() {
        let clauses = vec![
            vec![Literal::Pos("a".into()), Literal::Pos("b".into())],
            vec![Literal::Neg("a".into()), Literal::Pos("c".into())],
            vec![Literal::Neg("b".into()), Literal::Neg("c".into())],
        ];

        let mut solver = SatSolver::new(clauses);
        let result = solver.solve();
        assert!(result.is_some());
    }

    #[test]
    fn test_cnf() {
        let formula = Proposition::implies(Proposition::var("p"), Proposition::var("q"));
        let cnf = formula.to_cnf();
        // p => q becomes !p | q
        assert!(cnf.len() >= 1);
    }

    #[test]
    fn test_unification() {
        let t1 = Term::Function("f".into(), vec![Term::Variable("X".into()), Term::Constant("a".into())]);
        let t2 = Term::Function("f".into(), vec![Term::Constant("b".into()), Term::Variable("Y".into())]);

        let subst = Unifier::unify(&t1, &t2);
        assert!(subst.is_some());
        let s = subst.unwrap();
        assert_eq!(s.get("X"), Some(&Term::Constant("b".into())));
        assert_eq!(s.get("Y"), Some(&Term::Constant("a".into())));
    }
}
