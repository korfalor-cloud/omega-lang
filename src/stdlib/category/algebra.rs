/// Category theory: categories, functors, natural transformations, monads.

use std::collections::HashMap;

/// A category with objects and morphisms.
pub trait Category {
    type Object;
    type Morphism;

    fn identity(&self, obj: &Self::Object) -> Self::Morphism;
    fn compose(&self, f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism>;
    fn source(&self, m: &Self::Morphism) -> Self::Object;
    fn target(&self, m: &Self::Morphism) -> Self::Object;
}

/// FinSet: category of finite sets and functions.
#[derive(Debug, Clone)]
pub struct FinSet {
    pub sets: HashMap<usize, Vec<usize>>,
    pub functions: HashMap<usize, (usize, usize, Vec<(usize, usize)>)>, // id -> (dom, codom, pairs)
    next_id: usize,
}

impl FinSet {
    pub fn new() -> Self {
        Self {
            sets: HashMap::new(),
            functions: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn add_set(&mut self, elements: Vec<usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.sets.insert(id, elements);
        id
    }

    pub fn add_function(&mut self, dom: usize, codom: usize, mapping: Vec<(usize, usize)>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.functions.insert(id, (dom, codom, mapping));
        id
    }

    pub fn apply(&self, func_id: usize, input: usize) -> Option<usize> {
        self.functions.get(&func_id)
            .and_then(|(_, _, pairs)| pairs.iter().find(|&&(x, _)| x == input))
            .map(|&(_, y)| y)
    }

    /// Check if a function is injective (one-to-one).
    pub fn is_injective(&self, func_id: usize) -> bool {
        if let Some((_, _, pairs)) = self.functions.get(&func_id) {
            let mut seen = std::collections::HashSet::new();
            for &(_, y) in pairs {
                if !seen.insert(y) { return false; }
            }
            true
        } else {
            false
        }
    }

    /// Check if a function is surjective (onto).
    pub fn is_surjective(&self, func_id: usize) -> bool {
        if let Some((_, codom, pairs)) = self.functions.get(&func_id) {
            if let Some(target) = self.sets.get(codom) {
                let image: std::collections::HashSet<usize> = pairs.iter().map(|&(_, y)| y).collect();
                target.iter().all(|e| image.contains(e))
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Check if a function is bijective.
    pub fn is_bijective(&self, func_id: usize) -> bool {
        self.is_injective(func_id) && self.is_surjective(func_id)
    }

    /// Compute the product set A x B.
    pub fn product(&self, a: usize, b: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        if let (Some(set_a), Some(set_b)) = (self.sets.get(&a), self.sets.get(&b)) {
            for &x in set_a {
                for &y in set_b {
                    result.push((x, y));
                }
            }
        }
        result
    }

    /// Compute the coproduct (disjoint union) A + B.
    pub fn coproduct(&self, a: usize, b: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        if let Some(set_a) = self.sets.get(&a) {
            for &x in set_a {
                result.push((0, x)); // tag 0 = left
            }
        }
        if let Some(set_b) = self.sets.get(&b) {
            for &x in set_b {
                result.push((1, x)); // tag 1 = right
            }
        }
        result
    }
}

/// A functor between categories.
pub trait Functor<C1: Category, C2: Category> {
    fn map_object(&self, obj: &C1::Object) -> C2::Object;
    fn map_morphism(&self, m: &C1::Morphism) -> C2::Morphism;
}

/// A natural transformation between functors.
pub trait NaturalTransformation<C1: Category, C2: Category> {
    fn component(&self, obj: &C1::Object) -> C2::Morphism;
    fn naturality_check(&self, m: &C1::Morphism) -> bool;
}

/// Partial order as a category.
#[derive(Debug, Clone)]
pub struct PartialOrder {
    pub elements: Vec<usize>,
    pub order: HashMap<(usize, usize), bool>, // (a, b) => a <= b
}

impl PartialOrder {
    pub fn new(elements: Vec<usize>) -> Self {
        Self {
            elements,
            order: HashMap::new(),
        }
    }

    pub fn set_leq(&mut self, a: usize, b: usize) {
        self.order.insert((a, b), true);
    }

    pub fn is_leq(&self, a: usize, b: usize) -> bool {
        self.order.get(&(a, b)).copied().unwrap_or(false)
    }

    /// Check transitivity.
    pub fn is_transitive(&self) -> bool {
        for &a in &self.elements {
            for &b in &self.elements {
                for &c in &self.elements {
                    if self.is_leq(a, b) && self.is_leq(b, c) && !self.is_leq(a, c) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check antisymmetry.
    pub fn is_antisymmetric(&self) -> bool {
        for &a in &self.elements {
            for &b in &self.elements {
                if a != b && self.is_leq(a, b) && self.is_leq(b, a) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the meet (greatest lower bound) of two elements.
    pub fn meet(&self, a: usize, b: usize) -> Option<usize> {
        self.elements.iter()
            .filter(|&&x| self.is_leq(x, a) && self.is_leq(x, b))
            .max_by_key(|&&x| {
                self.elements.iter().filter(|&&y| self.is_leq(x, y)).count()
            })
            .copied()
    }

    /// Compute the join (least upper bound) of two elements.
    pub fn join(&self, a: usize, b: usize) -> Option<usize> {
        self.elements.iter()
            .filter(|&&x| self.is_leq(a, x) && self.is_leq(b, x))
            .min_by_key(|&&x| {
                self.elements.iter().filter(|&&y| self.is_leq(y, x)).count()
            })
            .copied()
    }

    /// Top element (if exists).
    pub fn top(&self) -> Option<usize> {
        self.elements.iter()
            .find(|&&x| self.elements.iter().all(|&y| self.is_leq(y, x)))
            .copied()
    }

    /// Bottom element (if exists).
    pub fn bottom(&self) -> Option<usize> {
        self.elements.iter()
            .find(|&&x| self.elements.iter().all(|&y| self.is_leq(x, y)))
            .copied()
    }
}

/// Lattice: a partial order where every pair has a meet and join.
pub struct Lattice {
    pub order: PartialOrder,
}

impl Lattice {
    pub fn new(order: PartialOrder) -> Self {
        Self { order }
    }

    pub fn is_lattice(&self) -> bool {
        for &a in &self.order.elements {
            for &b in &self.order.elements {
                if self.order.meet(a, b).is_none() || self.order.join(a, b).is_none() {
                    return false;
                }
            }
        }
        true
    }

    /// Check if it's a distributive lattice.
    pub fn is_distributive(&self) -> bool {
        for &a in &self.order.elements {
            for &b in &self.order.elements {
                for &c in &self.order.elements {
                    // a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
                    if let (Some(bjc), Some(ab), Some(ac)) = (self.order.join(b, c), self.order.meet(a, b), self.order.meet(a, c)) {
                        if let (Some(lhs), Some(rhs)) = (self.order.meet(a, bjc), self.order.join(ab, ac)) {
                            if lhs != rhs { return false; }
                        }
                    }
                }
            }
        }
        true
    }
}

/// A monoid: a set with an associative binary operation and identity.
pub trait Monoid {
    fn identity(&self) -> Self::Element;
    fn op(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    type Element: Clone;
}

/// Addition monoid on integers.
pub struct AdditiveInt;

impl Monoid for AdditiveInt {
    type Element = i64;
    fn identity(&self) -> i64 { 0 }
    fn op(&self, a: &i64, b: &i64) -> i64 { a + b }
}

/// Multiplication monoid on integers.
pub struct MultiplicativeInt;

impl Monoid for MultiplicativeInt {
    type Element = i64;
    fn identity(&self) -> i64 { 1 }
    fn op(&self, a: &i64, b: &i64) -> i64 { a * b }
}

/// String concatenation monoid.
pub struct StringConcat;

impl Monoid for StringConcat {
    type Element = String;
    fn identity(&self) -> String { String::new() }
    fn op(&self, a: &String, b: &String) -> String { format!("{}{}", a, b) }
}

/// Fold over a monoid.
pub fn monoid_fold<M: Monoid>(monoid: &M, elements: &[M::Element]) -> M::Element {
    elements.iter().fold(monoid.identity(), |acc, e| monoid.op(&acc, e))
}

/// Group: a monoid where every element has an inverse.
pub trait Group: Monoid {
    fn inverse(&self, a: &Self::Element) -> Self::Element;
}

/// Integers under addition form a group.
pub struct AdditiveGroup;

impl Monoid for AdditiveGroup {
    type Element = f64;
    fn identity(&self) -> f64 { 0.0 }
    fn op(&self, a: &f64, b: &f64) -> f64 { a + b }
}

impl Group for AdditiveGroup {
    fn inverse(&self, a: &f64) -> f64 { -a }
}

/// Symmetric group S_n (permutations of n elements).
pub struct SymmetricGroup {
    pub n: usize,
}

impl SymmetricGroup {
    pub fn new(n: usize) -> Self { Self { n } }

    pub fn identity_perm(&self) -> Vec<usize> {
        (0..self.n).collect()
    }

    pub fn compose(&self, p: &[usize], q: &[usize]) -> Vec<usize> {
        (0..self.n).map(|i| p[q[i]]).collect()
    }

    pub fn inverse(&self, p: &[usize]) -> Vec<usize> {
        let mut inv = vec![0; self.n];
        for i in 0..self.n {
            inv[p[i]] = i;
        }
        inv
    }

    pub fn is_valid_permutation(&self, p: &[usize]) -> bool {
        if p.len() != self.n { return false; }
        let mut seen = vec![false; self.n];
        for &x in p {
            if x >= self.n || seen[x] { return false; }
            seen[x] = true;
        }
        true
    }

    /// Cycle decomposition.
    pub fn cycles(&self, p: &[usize]) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.n];
        let mut cycles = Vec::new();

        for start in 0..self.n {
            if visited[start] { continue; }
            let mut cycle = Vec::new();
            let mut current = start;
            while !visited[current] {
                visited[current] = true;
                cycle.push(current);
                current = p[current];
            }
            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }
        cycles
    }

    /// Sign of a permutation (+1 or -1).
    pub fn sign(&self, p: &[usize]) -> i32 {
        let cycles = self.cycles(p);
        let num_transpositions: usize = cycles.iter().map(|c| c.len() - 1).sum();
        if num_transpositions % 2 == 0 { 1 } else { -1 }
    }

    /// Generate all permutations (for small n).
    pub fn all_permutations(&self) -> Vec<Vec<usize>> {
        let mut perms = Vec::new();
        let mut current: Vec<usize> = (0..self.n).collect();
        self.heap_permute(self.n, &mut current, &mut perms);
        perms
    }

    fn heap_permute(&self, size: usize, current: &mut Vec<usize>, perms: &mut Vec<Vec<usize>>) {
        if size == 1 {
            perms.push(current.clone());
            return;
        }
        for i in 0..size {
            self.heap_permute(size - 1, current, perms);
            if size % 2 == 0 {
                current.swap(i, size - 1);
            } else {
                current.swap(0, size - 1);
            }
        }
    }
}

/// Ring: a set with two operations (addition and multiplication).
pub trait Ring {
    type Element: Clone;
    fn zero(&self) -> Self::Element;
    fn one(&self) -> Self::Element;
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn neg(&self, a: &Self::Element) -> Self::Element;
}

/// Integers modulo n.
pub struct Zn {
    pub n: i64,
}

impl Ring for Zn {
    type Element = i64;
    fn zero(&self) -> i64 { 0 }
    fn one(&self) -> i64 { 1 }
    fn add(&self, a: &i64, b: &i64) -> i64 { (a + b) % self.n }
    fn mul(&self, a: &i64, b: &i64) -> i64 { (a * b) % self.n }
    fn neg(&self, a: &i64) -> i64 { (self.n - a) % self.n }
}

/// Polynomial ring R[x].
pub struct PolynomialRing {
    pub coefficients: Vec<f64>,
}

impl PolynomialRing {
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self { coefficients }
    }

    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        self.coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }

    pub fn add(&self, other: &PolynomialRing) -> PolynomialRing {
        let len = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; len];
        for i in 0..len {
            let a = self.coefficients.get(i).copied().unwrap_or(0.0);
            let b = other.coefficients.get(i).copied().unwrap_or(0.0);
            result[i] = a + b;
        }
        PolynomialRing::new(result)
    }

    pub fn multiply(&self, other: &PolynomialRing) -> PolynomialRing {
        let len = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; len];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        PolynomialRing::new(result)
    }

    pub fn derivative(&self) -> PolynomialRing {
        if self.coefficients.len() <= 1 {
            return PolynomialRing::new(vec![0.0]);
        }
        let result: Vec<f64> = self.coefficients.iter().enumerate()
            .skip(1)
            .map(|(i, &c)| i as f64 * c)
            .collect();
        PolynomialRing::new(result)
    }

    pub fn integral(&self) -> PolynomialRing {
        let mut result = vec![0.0];
        for (i, &c) in self.coefficients.iter().enumerate() {
            result.push(c / (i + 1) as f64);
        }
        PolynomialRing::new(result)
    }
}

/// Free group on generators.
pub struct FreeGroup {
    pub generators: Vec<char>,
}

impl FreeGroup {
    pub fn new(generators: Vec<char>) -> Self {
        Self { generators }
    }

    /// Reduce a word by cancelling adjacent inverses.
    pub fn reduce(&self, word: &[char]) -> Vec<char> {
        let mut stack: Vec<char> = Vec::new();
        for &c in word {
            if let Some(&top) = stack.last() {
                if self.are_inverse(top, c) {
                    stack.pop();
                    continue;
                }
            }
            stack.push(c);
        }
        stack
    }

    fn are_inverse(&self, a: char, b: char) -> bool {
        (a.is_uppercase() && a.to_lowercase().next() == Some(b))
            || (b.is_uppercase() && b.to_lowercase().next() == Some(a))
    }

    /// Compute the inverse of a word.
    pub fn inverse(&self, word: &[char]) -> Vec<char> {
        word.iter().rev().map(|&c| {
            if c.is_uppercase() { c.to_lowercase().next().unwrap() }
            else { c.to_uppercase().next().unwrap() }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finset() {
        let mut fs = FinSet::new();
        let a = fs.add_set(vec![1, 2, 3]);
        let b = fs.add_set(vec![4, 5, 6]);
        let f = fs.add_function(a, b, vec![(1, 4), (2, 5), (3, 6)]);

        assert!(fs.is_bijective(f));
        assert_eq!(fs.apply(f, 2), Some(5));
    }

    #[test]
    fn test_partial_order() {
        let mut po = PartialOrder::new(vec![1, 2, 3, 4, 6, 12]);
        // Divisibility order
        for &a in &po.elements.clone() {
            for &b in &po.elements.clone() {
                if b % a == 0 {
                    po.set_leq(a, b);
                }
            }
        }

        assert!(po.is_transitive());
        assert!(po.is_antisymmetric());
        assert_eq!(po.meet(4, 6), Some(2));
        assert_eq!(po.join(4, 6), Some(12));
    }

    #[test]
    fn test_symmetric_group() {
        let s3 = SymmetricGroup::new(3);
        let p = vec![1, 2, 0]; // (0 1 2)
        let q = vec![1, 0, 2]; // (0 1)

        assert!(s3.is_valid_permutation(&p));
        let composed = s3.compose(&p, &q);
        assert_eq!(composed, vec![2, 1, 0]);

        let cycles = s3.cycles(&p);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_polynomial() {
        let p = PolynomialRing::new(vec![1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
        assert_eq!(p.degree(), 2);
        assert_eq!(p.evaluate(2.0), 1.0 + 4.0 + 12.0);

        let dp = p.derivative();
        assert_eq!(dp.coefficients, vec![2.0, 6.0]);
    }

    #[test]
    fn test_free_group() {
        let fg = FreeGroup::new(vec!['a', 'b']);
        let word = vec!['a', 'A', 'b', 'B', 'a'];
        let reduced = fg.reduce(&word);
        assert_eq!(reduced, vec!['a']);
    }
}
