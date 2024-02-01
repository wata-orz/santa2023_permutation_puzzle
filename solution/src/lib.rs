use proconio::input;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    ops::*,
};
use tools::parse_output;

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { vec![$($e),*] };
	($($e:expr,)*) => { vec![$($e),*] };
	($e:expr; $d:expr) => { vec![$e; $d] };
	($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        // ローカル環境とジャッジ環境の実行速度差はget_timeで吸収しておくと便利
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 1.0
        }
        #[cfg(not(feature = "local"))]
        {
            ms - STIME
        }
    }
}

#[allow(non_snake_case)]

pub struct OnDrop<F: Fn()> {
    f: F,
}

impl<F: Fn()> OnDrop<F> {
    #[inline]
    pub fn new(f: F) -> Self {
        OnDrop { f }
    }
}

impl<F: Fn()> Drop for OnDrop<F> {
    #[inline]
    fn drop(&mut self) {
        (self.f)()
    }
}

pub fn bench(id: String) -> OnDrop<impl Fn()> {
    eprintln!("Start({})", id);
    let t = ::std::time::SystemTime::now();
    OnDrop::new(move || {
        let d = t.elapsed().unwrap();
        let s = d.as_secs() as f64 + d.subsec_nanos() as f64 * 1e-9;
        eprintln!("Time({}) = {:.3}", id, s);
    })
}

#[macro_export]
macro_rules! bench {
	([$name:expr]$($e: tt)*) => {
		let b = $crate::bench($name.to_owned());
		$($e)*
		drop(b);
	};
	($($e: tt)*) => {
		let b = $crate::bench(format!("{}:{}", file!(), line!()));
		$($e)*
		drop(b);
	};
}

pub static mut PROFILER: *mut Vec<(&str, &(f64, usize, usize))> = 0 as *mut Vec<_>;

#[macro_export]
macro_rules! profile {
    ($id:ident) => {
        static mut __PROF: (f64, usize, usize) = (0.0, 0, 0);
        unsafe {
            if __PROF.1 == 0 {
                if $crate::PROFILER.is_null() {
                    $crate::PROFILER = Box::into_raw(Box::new(Vec::new()));
                }
                (*$crate::PROFILER).push((stringify!($id), &__PROF));
            }
            if __PROF.2 == 0 {
                let d = ::std::time::SystemTime::now()
                    .duration_since(::std::time::SystemTime::UNIX_EPOCH)
                    .unwrap();
                __PROF.0 -= d.as_secs() as f64 + d.subsec_nanos() as f64 * 1e-9;
            }
            __PROF.1 += 1;
            __PROF.2 += 1;
        }
        #[allow(unused)]
        let $id = $crate::OnDrop::new(move || unsafe {
            __PROF.2 -= 1;
            if __PROF.2 == 0 {
                let d = ::std::time::SystemTime::now()
                    .duration_since(::std::time::SystemTime::UNIX_EPOCH)
                    .unwrap();
                __PROF.0 += d.as_secs() as f64 + d.subsec_nanos() as f64 * 1e-9;
            }
        });
    };
}

#[macro_export]
macro_rules! count {
    ($id:ident) => {
        static mut __PROF: (f64, usize, usize) = (0.0, 0, 0);
        unsafe {
            if __PROF.1 == 0 {
                if $crate::PROFILER.is_null() {
                    $crate::PROFILER = Box::into_raw(Box::new(Vec::new()));
                }
                (*$crate::PROFILER).push((stringify!($id), &__PROF));
            }
            __PROF.1 += 1;
        }
    };
}

pub fn write_profile() {
    let mut ps: Vec<_> = unsafe {
        if PROFILER.is_null() {
            return;
        }
        (*PROFILER).clone()
    };
    ps.sort_by(|&(_, a), &(_, b)| b.partial_cmp(&a).unwrap());
    eprintln!("########## Profile ##########");
    for (id, &(mut t, c, depth)) in ps {
        if depth > 0 {
            let d = ::std::time::SystemTime::now()
                .duration_since(::std::time::SystemTime::UNIX_EPOCH)
                .unwrap();
            t += d.as_secs() as f64 + d.subsec_nanos() as f64 * 1e-9;
        }
        eprintln!("{}:\t{:.3}\t{}", id, t, c);
    }
    eprintln!("#############################");
}

#[macro_export]
macro_rules! optuna {
	($($p:ident: $t:tt = suggest($def:expr, $($a:tt),*)),* $(,)*) => {
		#[derive(Debug, Clone)]
		struct Param {
			$($p: $t,)*
		}
		lazy_static::lazy_static! {
			static ref PARAM: Param = {
				$(let $p = std::env::var(stringify!($p)).map(|s| s.parse().expect(concat!("failed to parse ", stringify!($p)))).unwrap_or($def);)*
				Param { $( $p ),* }
			};
		}
		impl Param {
			fn optuna_str() -> String {
				let mut list = vec![];
				$(list.push($crate::optuna!(# $p, $t, $($a),*));)*
				let mut s = "def setup(trial):\n".to_owned();
				for (t, _) in &list {
					s += "\t";
					s += &t;
					s += "\n";
				}
				s += "\tenv = {";
				for (i, (_, t)) in list.iter().enumerate() {
					if i > 0 {
						s += ", ";
					}
					s += &t;
				}
				s += "}\n\treturn env";
				s
			}
		}
	};
	(# $p:ident, f64, $min:expr, $max:expr) => {
		(format!("{} = trial.suggest_float(\"{}\", {}, {})", stringify!($p), stringify!($p), $min, $max), format!("\"{}\": str({})", stringify!($p), stringify!($p)))
	};
	(# $p:ident, usize, $min:expr, $max:expr) => {
		(format!("{} = trial.suggest_int(\"{}\", {}, {})", stringify!($p), stringify!($p), $min, $max), format!("\"{}\": str({})", stringify!($p), stringify!($p)))
	};
	(# $p:ident, f64, $min:expr, $max:expr, log) => {
		(format!("{} = trial.suggest_float(\"{}\", {}, {}, log=True)", stringify!($p), stringify!($p), $min, $max), format!("\"{}\": str({})", stringify!($p), stringify!($p)))
	};
	(# $p:ident, f64, $min:expr, $max:expr, $step:expr) => {
		(format!("{} = trial.suggest_float(\"{}\", {}, {}, {})", stringify!($p), stringify!($p), $min, $max, $step), format!("\"{}\": str({})", stringify!($p), stringify!($p)))
	};
}

pub struct Trace<T: Copy> {
    log: Vec<(T, u32)>,
}

impl<T: Copy> Trace<T> {
    pub fn new() -> Self {
        Trace { log: vec![] }
    }
    pub fn add(&mut self, c: T, p: u32) -> u32 {
        self.log.push((c, p));
        self.log.len() as u32 - 1
    }
    pub fn get(&self, mut i: u32) -> Vec<T> {
        let mut out = vec![];
        while i != !0 {
            out.push(self.log[i as usize].0);
            i = self.log[i as usize].1;
        }
        out.reverse();
        out
    }
    pub fn prev(&self, p: u32) -> u32 {
        self.log[p as usize].1
    }
    pub fn prev_move(&self, p: u32) -> T {
        self.log[p as usize].0
    }
}

use itertools::Itertools;
use tools::*;

pub fn read_input() -> Input {
    input! {
        id: usize, puzzle_type: String, n: usize, m: usize, wildcard: usize,
        moves: [(String, [usize; n]); m],
        target: [u16; n],
        start: [u16; n],
    }
    Input {
        id,
        puzzle_type,
        n,
        m,
        moves: moves
            .iter()
            .map(|(_, v)| v.clone())
            .chain(moves.iter().map(|(_, v)| inv_perm(&v)))
            .collect(),
        move_names: moves
            .iter()
            .map(|(v, _)| v.clone())
            .chain(moves.iter().map(|(v, _)| format!("-{}", v)))
            .collect(),
        target,
        start,
        wildcard,
    }
}

pub fn read_best(input: &Input) -> Vec<usize> {
    let base = env!("CARGO_MANIFEST_DIR");
    let f = std::fs::read_to_string(format!("{}/all/best/{:04}.txt", base, input.id)).unwrap();
    parse_output(input, &f).unwrap().out
}

pub fn write_output(input: &Input, out: &[usize]) {
    println!("{}", out.iter().map(|&i| &input.move_names[i]).join("."));
}

pub fn apply<T: Copy, I: Copy>(crt: &[T], perm: &[I]) -> Vec<T>
where
    usize: From<I>,
{
    perm.iter().map(|&p| crt[usize::from(p)]).collect()
}

pub fn diff(crt: &[u16], target: &[u16]) -> usize {
    crt.iter().zip(target.iter()).filter(|(a, b)| a != b).count()
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Mat<T>(pub Vec<Vec<T>>);

impl<T> Mat<T>
where
    T: Clone + Default,
{
    pub fn row(a: Vec<T>) -> Mat<T> {
        Mat(vec![a])
    }
    pub fn col(a: Vec<T>) -> Mat<T> {
        Mat(a.into_iter().map(|v| vec![v]).collect())
    }
    pub fn to_vec(&self) -> Vec<T> {
        if self.len() == 1 {
            self[0].clone()
        } else if self[0].len() == 1 {
            self.iter().map(|r| r[0].clone()).collect()
        } else {
            panic!("Cannot transform Mat into Vec. row: {}, col: {}", self.len(), self[0].len());
        }
    }
    pub fn transpose(&self) -> Mat<T> {
        let mut v = vec![Vec::with_capacity(self.len()); self[0].len()];
        for r in self.iter() {
            for (i, x) in r.iter().enumerate() {
                v[i].push(x.clone());
            }
        }
        Mat(v)
    }
    /// O(nmr), r: rank.
    pub fn gauss(self) -> Decomposed<T>
    where
        T: Clone + Default + PartialOrd + From<u8> + SubAssign,
        for<'b> &'b T: Neg<Output = T> + Mul<Output = T> + Div<Output = T>,
    {
        let mut a = self;
        let (n, m) = (a.len(), a[0].len());
        let (zero, one) = (T::default(), T::from(1));
        let abs = |a: &T| {
            if a < &zero {
                a.neg()
            } else {
                a.clone()
            }
        };
        let mut pivots = vec![n; m];
        let mut r = 0;
        for c in 0..m {
            let mut t = abs(&a[r][c]);
            let mut p = r;
            for i in r + 1..n {
                if t.setmax(abs(&a[i][c])) {
                    p = i;
                }
            }
            if t == zero {
                continue;
            }
            a.swap(r, p);
            pivots[c] = p;
            let inv = &one / &a[r][c];
            for arj in &mut a[r][c + 1..] {
                *arj = &*arj * &inv
            }
            let (ar, a) = a[r..].split_first_mut().unwrap();
            for ai in a {
                let d = ai[c].clone();
                for (aij, arj) in ai[c + 1..].iter_mut().zip(ar[c + 1..].iter()) {
                    *aij -= &d * arj;
                }
            }
            r += 1;
            if r == n {
                break;
            }
        }
        Decomposed(a, pivots)
    }
}

#[derive(Clone, Debug)]
pub struct Decomposed<T>(pub Mat<T>, pub Vec<usize>);

impl<T> Decomposed<T>
where
    T: Clone + Default + PartialOrd + From<u8> + SubAssign,
    for<'b> &'b T: Neg<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    pub fn rank(&self) -> usize {
        let n = self.0.len();
        self.1.iter().filter(|&&x| x < n).count()
    }
    pub fn det(&self) -> T {
        let Decomposed(ref a, ref pivots) = *self;
        assert_eq!(a.len(), a[0].len());
        let mut det = T::from(1);
        for i in 0..a.len() {
            if pivots[i] == a.len() {
                return T::default();
            }
            det = &det * &a[i][i];
        }
        det
    }
    /// Solve Ax=b. x={x[0]+linearspace(x[1..])}.
    /// O(nr+(1+m-r)r^2 ), r: rank.
    pub fn solve(&self, mut b: Vec<T>) -> Vec<Vec<T>>
    where
        T: ::std::fmt::Debug,
    {
        let Decomposed(ref a, ref pivots) = *self;
        let (n, m) = (a.len(), a[0].len());
        assert_eq!(n, b.len());
        let (zero, one) = (T::default(), T::from(1));
        let mut id = vec![m; n + 1];
        let mut r = 0;
        for c in 0..m {
            if pivots[c] == n {
                continue;
            }
            b.swap(r, pivots[c]);
            id[r] = c;
            r += 1;
            if r == n {
                break;
            }
        }
        for r in 0..r {
            let c = id[r];
            b[r] = &b[r] / &a[r][c];
            for i in r + 1..n {
                let tmp = &a[i][c] * &b[r];
                b[i] -= tmp;
            }
        }
        for r in (1..r).rev() {
            let c = id[r];
            for i in 0..r {
                let tmp = &a[i][c] * &b[r];
                b[i] -= tmp;
            }
        }
        if b[r..].iter().any(|v| v != &zero) {
            return vec![];
        }
        let mut x = mat![T::default(); 1 + m - r; m];
        let mut k = 0;
        for j in 0..m {
            if id[k] == j {
                x[0][j] = b[k].clone();
                k += 1;
            } else {
                let mut c: Vec<T> = a[0..k].iter().map(|ai| ai[j].neg()).collect();
                for r in (1..k).rev() {
                    for i in 0..r {
                        let tmp = &a[i][id[r]] * &c[r];
                        c[i] -= tmp;
                    }
                }
                for (i, v) in c.into_iter().enumerate() {
                    x[1 + j - k][id[i]] = v;
                }
                x[1 + j - k][j] = one.clone();
            }
        }
        x
    }
}

impl<T> Deref for Mat<T> {
    type Target = Vec<Vec<T>>;
    #[inline]
    fn deref(&self) -> &Vec<Vec<T>> {
        &self.0
    }
}

impl<T> DerefMut for Mat<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<Vec<T>> {
        &mut self.0
    }
}

pub trait Int {
    fn as_i32(self) -> i32;
    fn as_i64(self) -> i64;
    fn as_u32(self) -> u32;
    fn as_u64(self) -> u64;
    fn as_usize(self) -> usize;
}

macro_rules! impl_int {
	() => {};
	($t:ty $(, $ts:ty)*) => {
		impl Int for $t {
			fn as_i32(self) -> i32 {
				self as i32
			}
			fn as_i64(self) -> i64 {
				self as i64
			}
			fn as_u32(self) -> u32 {
				self as u32
			}
			fn as_u64(self) -> u64 {
				self as u64
			}
			fn as_usize(self) -> usize {
				self as usize
			}
		}
		impl_int!($($ts),*);
	}
}

impl_int!(i8, i16, i32, i64, u8, u16, u32, u64, isize, usize);

use std::marker::PhantomData;

pub trait Mod: Copy + Default + Ord {
    fn m() -> u32;
}

#[derive(Clone, Copy, Default, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub struct ModP<M>(pub u32, PhantomData<M>);

impl<M> ::std::fmt::Debug for ModP<M> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<M> ::std::fmt::Display for ModP<M> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<M: Mod> ModP<M> {
    /// Assume 0 <= a < m.
    pub fn new<I: Int>(a: I) -> ModP<M> {
        ModP(a.as_u32(), PhantomData)
    }
    pub fn pow<I: Int>(&self, b: I) -> ModP<M> {
        let mut b = b.as_i64();
        if b < 0 {
            b = b % (M::m() - 1) as i64 + M::m() as i64 - 1;
        }
        let mut res = ModP::new(1);
        let mut a = *self;
        while b > 0 {
            if (b & 1) > 0 {
                res *= a;
            }
            a *= a;
            b >>= 1;
        }
        res
    }
    pub fn inv(&self) -> ModP<M> {
        self.pow(M::m() as u64 - 2)
    }
}

impl<'a, M: Mod> Add for &'a ModP<M> {
    type Output = ModP<M>;
    fn add(self, a: &ModP<M>) -> ModP<M> {
        let v = self.0 + a.0;
        if v >= M::m() {
            ModP::new(v - M::m())
        } else {
            ModP::new(v)
        }
    }
}

impl<'a, M: Mod> Sub for &'a ModP<M> {
    type Output = ModP<M>;
    fn sub(self, a: &ModP<M>) -> ModP<M> {
        if self.0 < a.0 {
            ModP::new(M::m() - a.0 + self.0)
        } else {
            ModP::new(self.0 - a.0)
        }
    }
}

impl<'a, M: Mod> Mul for &'a ModP<M> {
    type Output = ModP<M>;
    fn mul(self, a: &ModP<M>) -> ModP<M> {
        ModP::new(((self.0 as u64 * a.0 as u64) % M::m() as u64) as u32)
    }
}

impl<'a, M: Mod> Div for &'a ModP<M> {
    type Output = ModP<M>;
    fn div(self, a: &ModP<M>) -> ModP<M> {
        self * a.inv()
    }
}

impl<'a, M: Mod> Neg for &'a ModP<M> {
    type Output = ModP<M>;
    fn neg(self) -> ModP<M> {
        ModP::new(if self.0 == 0 { 0 } else { M::m() - self.0 })
    }
}

impl<M: Mod> Neg for ModP<M> {
    type Output = ModP<M>;
    fn neg(self) -> ModP<M> {
        (&self).neg()
    }
}

macro_rules! impl_all {
	($t:ident$(<$($g:ident),*>)*; $Op:ident:$op:ident:$Opa:ident:$opa:ident) => {
		impl<$($($g),*)*> $Op for $t$(<$($g),*>)* where for<'b> &'b $t$(<$($g),*>)*: $Op<Output = $t$(<$($g),*>)*> {
			type Output = $t$(<$($g),*>)*;
			fn $op(self, a: $t$(<$($g),*>)*) -> $t$(<$($g),*>)* { (&self).$op(&a) }
		}
		impl<'a, $($($g),*)*> $Op<&'a $t$(<$($g),*>)*> for $t$(<$($g),*>)* where for<'b> &'b $t$(<$($g),*>)*: $Op<Output = $t$(<$($g),*>)*> {
			type Output = $t$(<$($g),*>)*;
			fn $op(self, a: &$t$(<$($g),*>)*) -> $t$(<$($g),*>)* { (&self).$op(&a) }
		}
		impl<'a, $($($g),*)*> $Op<$t$(<$($g),*>)*> for &'a $t$(<$($g),*>)* where for<'b> &'b $t$(<$($g),*>)*: $Op<Output = $t$(<$($g),*>)*> {
			type Output = $t$(<$($g),*>)*;
			fn $op(self, a: $t$(<$($g),*>)*) -> $t$(<$($g),*>)* { (&self).$op(&a) }
		}
		impl<$($($g),*)*> $Opa for $t$(<$($g),*>)* where for<'b> &'b $t$(<$($g),*>)*: $Op<Output = $t$(<$($g),*>)*> {
			fn $opa(&mut self, a: $t$(<$($g),*>)*) { *self = (&*self).$op(&a) }
		}
	}
}

impl_all!(ModP<M>; Add:add:AddAssign:add_assign);
impl_all!(ModP<M>; Sub:sub:SubAssign:sub_assign);
impl_all!(ModP<M>; Mul:mul:MulAssign:mul_assign);
impl_all!(ModP<M>; Div:div:DivAssign:div_assign);

#[macro_export]
macro_rules! define_mod {
    ($name:ident; $e:expr) => {
        #[derive(Clone, Copy, Default, Hash, PartialEq, PartialOrd, Eq, Ord)]
        pub struct _Mod;
        pub type $name = $crate::ModP<_Mod>;
        impl $crate::Mod for _Mod {
            fn m() -> u32 {
                $e
            }
        }
    };
}

impl<M: Mod, I: Int> From<I> for ModP<M> {
    /// Assume 0 <= a < m.
    fn from(a: I) -> ModP<M> {
        ModP::new(a)
    }
}

pub fn gcd(mut x: i64, mut y: i64) -> i64 {
    while y != 0 {
        let t = x % y;
        x = y;
        y = t
    }
    x
}

/// ax+by=gcd(x,y)となるような数の組{gcd,a,b}を一組求める．
/// (a,b)の一般解は(a+d*y/c,b-d*x/c)．
pub fn exgcd(x: i64, y: i64) -> [i64; 3] {
    let mut u = [x, 1, 0];
    let mut v = [y, 0, 1];
    while v[0] != 0 {
        let t = u[0] / v[0];
        for (i, j) in u.iter_mut().zip(v.iter_mut()) {
            *i -= *j * t;
        }
        ::std::mem::swap(&mut u, &mut v);
    }
    u
}

pub fn optimize_bisearch(input: &Input, out: &[usize]) -> Vec<usize> {
    eprintln!("{:.3}: {}", get_time(), out.len());
    let mut crt = input.start.clone();
    let mut trace = Trace::new();
    let mut ids = HashMap::new();
    let mut id = !0;
    for &i in out {
        // if diff(&crt, &input.target) <= input.wildcard {
        //     break;
        // }
        crt = apply(&crt, &input.moves[i]);
        if let Some(id2) = ids.get(&crt) {
            id = *id2;
        } else {
            id = trace.add(i, id);
            ids.insert(crt.clone(), id);
        }
    }
    let mut out = trace.get(id);
    loop {
        let mut i = 0;
        let mut ok = false;
        while i < out.len() {
            for j in i + 1..out.len() {
                if out[i] == out[j] + input.m || out[i] + input.m == out[j] {
                    out.remove(j);
                    out.remove(i);
                    ok = true;
                    break;
                }
                if (0..input.n).any(|k| input.moves[out[i]][k] != k && input.moves[out[j]][k] != k) {
                    break;
                }
            }
            i += 1;
        }
        if !ok {
            break;
        }
    }
    'lp: loop {
        eprintln!("{:.3}: {}", get_time(), out.len());
        let mut fwd = vec![input.start.clone()];
        for i in 0..out.len() {
            fwd.push(apply(&fwd[i], &input.moves[out[i]]));
        }
        let crt_diff = diff(&fwd[out.len()], &input.target);
        let mut bwd = (0..input.n).collect_vec();
        let mut best = (!0, !0);
        let mut best_rate = 0.0;
        for t in (0..out.len()).rev() {
            for d in 1..=t.min(200) {
                let new_diff = diff(&apply(&fwd[t - d], &bwd), &input.target);
                if new_diff <= input.wildcard {
                    if new_diff <= crt_diff {
                        out = out[..t - d].iter().chain(&out[t + 1..]).cloned().collect_vec();
                        continue 'lp;
                    } else if best_rate.setmax((d - 1) as f64 / (new_diff - crt_diff) as f64) {
                        best = (t, d);
                    }
                }
            }
            bwd = apply(&input.moves[out[t]], &bwd);
        }
        if best.0 == !0 {
            break;
        }
        let (t, d) = best;
        out = out[..t - d].iter().chain(&out[t + 1..]).cloned().collect_vec();
    }
    let mut visited = HashSet::new();
    let mut que = vec![];
    let mut trace = Trace::new();
    que.push(((0..input.n as u16).collect_vec(), 0, !0));
    let mut qs = 0;
    while qs < que.len() {
        if que.len() * input.n >= 10000000 {
            break;
        }
        let (crt, d, id) = que[qs].clone();
        qs += 1;
        for mv in 0..input.moves.len() {
            let next = apply(&crt, &input.moves[mv]);
            if visited.insert(next.clone()) {
                que.push((next, d + 1, trace.add(mv, id)));
            }
        }
    }
    let max_d = que.last().unwrap().1;
    eprintln!("!log max_d {}", max_d);
    // write_output(&input, &out);
    eprintln!("{:.3}: {}", get_time(), out.len());
    loop {
        let mut ok = false;
        let mut i = 0;
        let mut crt = input.start.clone();
        while i < out.len() {
            if i * 100 / out.len() < (i + 1) * 100 / out.len() {
                eprintln!("{} / {}", i, out.len());
            }
            let k = (i + max_d * 2 - 1).min(out.len());
            let mut fwd = HashMap::new();
            let mut min = k - i;
            for &(ref mv, d, id) in &que {
                let f = apply(&crt, mv);
                if diff(&f, &input.target) <= input.wildcard {
                    if min.setmin(d) {
                        let replace = trace.get(id);
                        eprintln!(
                            "{} -> {}",
                            out[i..].iter().map(|&op| &input.move_names[op]).join("."),
                            replace.iter().map(|&op| &input.move_names[op]).join(".")
                        );
                        out = out[..i].iter().cloned().chain(replace).collect_vec();
                        // write_output(&input, &out);
                        eprintln!("{:.3}: {}", get_time(), out.len());
                        ok = true;
                    }
                }
                if !fwd.contains_key(&f) {
                    fwd.insert(f, (d, id));
                }
            }
            if min < k - i {
                continue;
            }
            let mut next = crt.clone();
            for j in i..k {
                next = apply(&next, &input.moves[out[j]]);
            }
            let mut min_id = (!0, !0);
            for &(ref mv, d, id) in &que {
                let b = apply(&next, mv);
                if let Some((d2, id2)) = fwd.get(&b) {
                    if min.setmin(d + d2) {
                        min_id = (id, *id2);
                    }
                }
            }
            if min < k - i {
                let mut replace = trace.get(min_id.1);
                replace.extend(
                    trace
                        .get(min_id.0)
                        .into_iter()
                        .rev()
                        .map(|i| if i < input.m { i + input.m } else { i - input.m }),
                );
                eprintln!(
                    "{} -> {}",
                    out[i..k].iter().map(|&op| &input.move_names[op]).join("."),
                    replace.iter().map(|&op| &input.move_names[op]).join(".")
                );
                out = out[..i]
                    .iter()
                    .cloned()
                    .chain(replace)
                    .chain(out[k..].iter().cloned())
                    .collect_vec();
                // write_output(&input, &out);
                eprintln!("{:.3}: {}", get_time(), out.len());
                ok = true;
            } else {
                crt = apply(&crt, &input.moves[out[i]]);
                i += 1;
            }
        }
        if !ok {
            break;
        }
    }
    eprintln!("Time = {:.3}", get_time());
    out
}

pub fn optimize_bisearch_globe(input: &Input, out: &[usize]) -> Vec<usize> {
    let mut out = out.to_vec();
    if !input.puzzle_type.starts_with("globe") {
        return out;
    }
    let rows = input.move_names.iter().filter(|s| s.starts_with("r")).count() as usize;
    let cols = input.move_names.iter().filter(|s| s.starts_with("f")).count() as usize;
    loop {
        let mut ok = false;
        for iter in 0..2 {
            for c in 0..cols / 2 {
                eprintln!("c: {} / {}", c, cols / 2);
                let mut moves = if iter == 0 {
                    vec![c]
                } else {
                    vec![
                        c,
                        c + 1,
                        (c + cols - 1) % cols,
                        c + cols / 2,
                        (c + cols / 2 + 1) % cols,
                        c + cols / 2 - 1,
                    ]
                };
                moves.extend(cols..cols + rows);
                moves.extend(cols + rows + cols..cols + rows + cols + rows);
                let mut visited = HashSet::new();
                let mut que = vec![];
                let mut trace = Trace::new();
                que.push(((0..input.n as u16).collect_vec(), 0, !0));
                let mut qs = 0;
                while qs < que.len() {
                    if que.len() * input.n >= 10000000 {
                        break;
                    }
                    let (crt, d, id) = que[qs].clone();
                    qs += 1;
                    for &mv in &moves {
                        let next = apply(&crt, &input.moves[mv]);
                        if visited.insert(next.clone()) {
                            que.push((next, d + 1, trace.add(mv, id)));
                        }
                    }
                }
                let max_d = que.last().unwrap().1;
                eprintln!("!log max_d {}", max_d);
                // write_output(&input, &out);
                eprintln!("{:.3}: {}", get_time(), out.len());
                let mut i = 0;
                let mut crt = input.start.clone();
                while i < out.len() {
                    if i * 100 / out.len() < (i + 1) * 100 / out.len() {
                        eprintln!("{} / {}", i, out.len());
                    }
                    let k = (i + max_d * 2 - 1).min(out.len());
                    let mut fwd = HashMap::new();
                    let mut min = k - i;
                    for &(ref mv, d, id) in &que {
                        let f = apply(&crt, mv);
                        if diff(&f, &input.target) <= input.wildcard {
                            if min.setmin(d) {
                                let replace = trace.get(id);
                                eprintln!(
                                    "{} -> {}",
                                    out[i..].iter().map(|&op| &input.move_names[op]).join("."),
                                    replace.iter().map(|&op| &input.move_names[op]).join(".")
                                );
                                out = out[..i].iter().cloned().chain(replace).collect_vec();
                                // write_output(&input, &out);
                                eprintln!("{:.3}: {}", get_time(), out.len());
                                ok = true;
                            }
                        }
                        if !fwd.contains_key(&f) {
                            fwd.insert(f, (d, id));
                        }
                    }
                    if min < k - i {
                        continue;
                    }
                    let mut next = crt.clone();
                    for j in i..k {
                        next = apply(&next, &input.moves[out[j]]);
                    }
                    let mut min_id = (!0, !0);
                    for &(ref mv, d, id) in &que {
                        let b = apply(&next, mv);
                        if let Some((d2, id2)) = fwd.get(&b) {
                            if min.setmin(d + d2) {
                                min_id = (id, *id2);
                            }
                        }
                    }
                    if min < k - i {
                        let mut replace = trace.get(min_id.1);
                        replace.extend(trace.get(min_id.0).into_iter().rev().map(|i| {
                            if i < input.m {
                                i + input.m
                            } else {
                                i - input.m
                            }
                        }));
                        eprintln!(
                            "{} -> {}",
                            out[i..k].iter().map(|&op| &input.move_names[op]).join("."),
                            replace.iter().map(|&op| &input.move_names[op]).join(".")
                        );
                        out = out[..i]
                            .iter()
                            .cloned()
                            .chain(replace)
                            .chain(out[k..].iter().cloned())
                            .collect_vec();
                        // write_output(&input, &out);
                        eprintln!("{:.3}: {}", get_time(), out.len());
                        ok = true;
                    } else {
                        crt = apply(&crt, &input.moves[out[i]]);
                        i += 1;
                    }
                }
            }
        }
        if !ok {
            break;
        }
    }
    eprintln!("Time = {:.3}", get_time());
    out
}

#[derive(Clone, Debug)]
struct Entry<K, V> {
    k: K,
    v: V,
}

impl<K: PartialOrd, V> Ord for Entry<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<K: PartialOrd, V> PartialOrd for Entry<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.k.partial_cmp(&other.k)
    }
}

impl<K: PartialEq, V> PartialEq for Entry<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.k.eq(&other.k)
    }
}

impl<K: PartialEq, V> Eq for Entry<K, V> {}

/// K が小さいトップn個を保持
#[derive(Clone, Debug)]
pub struct BoundedSortedList<K: PartialOrd + Copy, V: Clone> {
    que: BinaryHeap<Entry<K, V>>,
    size: usize,
}

impl<K: PartialOrd + Copy, V: Clone> BoundedSortedList<K, V> {
    pub fn new(size: usize) -> Self {
        Self {
            que: BinaryHeap::with_capacity(size),
            size,
        }
    }
    pub fn can_insert(&self, k: K) -> bool {
        self.que.len() < self.size || self.que.peek().unwrap().k > k
    }
    pub fn insert(&mut self, k: K, v: V) {
        if self.can_insert(k) {
            if self.que.len() == self.size {
                self.que.pop();
            }
            self.que.push(Entry { k, v });
        }
    }
    pub fn list(&self) -> Vec<(K, V)> {
        let mut v = self.que.clone().into_vec();
        v.sort();
        v.into_iter().map(|e| (e.k, e.v)).collect()
    }
    pub fn len(&self) -> usize {
        self.que.len()
    }
}

pub fn next_permutation<T>(a: &mut [T]) -> bool
where
    T: PartialOrd,
{
    let n = a.len();
    for i in (1..n).rev() {
        if a[i - 1] < a[i] {
            let mut j = n - 1;
            while a[i - 1] >= a[j] {
                j -= 1;
            }
            a.swap(i - 1, j);
            a[i..n].reverse();
            return true;
        }
    }
    a.reverse();
    false
}
