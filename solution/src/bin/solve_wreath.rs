#![allow(non_snake_case)]

use rand::{seq::SliceRandom, Rng};
use rand_core::SeedableRng;
use rayon::prelude::*;
use solution::*;
use std::sync::Mutex;
use tools::Input;

fn main() {
    get_time();
    let input = read_input();
    if !input.puzzle_type.starts_with("wreath") {
        return;
    }
    assert_eq!(input.m, 2);
    let mut loops = vec![];
    for k in 0..2 {
        for i in 0..input.n {
            if input.moves[k][i] != i {
                let mut list = vec![];
                let mut j = i;
                loop {
                    list.push(j);
                    j = input.moves[k][j];
                    if j == i {
                        break;
                    }
                }
                loops.push(list);
                break;
            }
        }
    }
    dbg!(loops);
    let out = beam(&input);
    write_output(&input, &out);
    eprintln!("Time = {:.3}", get_time());
}

#[derive(Clone, Debug)]
struct State {
    crt: Vec<u16>,
    diff: usize,
    ps: [usize; 2],
    id: usize,
}

impl State {
    fn new(crt: Vec<u16>, input: &Input, id: usize) -> Self {
        Self {
            diff: diff(&crt, &input.target),
            ps: [
                crt.iter().position(|&c| c == 0).unwrap(),
                crt.iter().rposition(|&c| c == 0).unwrap(),
            ],
            crt,
            id,
        }
    }
}

fn beam(input: &Input) -> Vec<usize> {
    let mut count = vec![0; input.n];
    for k in 0..2 {
        for i in 0..input.n {
            if input.moves[k][i] != i {
                count[i] += 1;
            }
        }
    }
    let trace = Mutex::new(Trace::new());
    let mut beam = vec![];
    let init = State::new(input.start.clone(), input, !0);
    beam.push(init);
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(849028);
    let mut hash = mat![0; input.n; 3];
    for i in 0..input.n {
        for j in 0..3 {
            hash[i][j] = rng.gen::<u64>();
        }
    }
    let visited = Set::new();
    visited.insert(input.start.iter().enumerate().fold(0, |h, (i, &c)| h ^ hash[i][c as usize]));
    for iter in 0..406 {
        eprintln!("{:.3}: {}", get_time(), iter);
        let mut list = vec![];
        beam.into_par_iter()
            .map(|s| {
                if s.diff <= input.wildcard {
                    let out = trace.lock().unwrap().get(s.id);
                    write_output(input, &out);
                    std::process::exit(0);
                }
                let mut list = vec![];
                for k in 0..input.moves.len() {
                    let crt = apply(&s.crt, &input.moves[k]);
                    if visited.insert(crt.iter().enumerate().fold(0, |h, (i, &c)| h ^ hash[i][c as usize])) {
                        let id = trace.lock().unwrap().add(k, s.id);
                        let state = State::new(crt, input, id);
                        list.push(state);
                    }
                }
                list
            })
            .collect_into_vec(&mut list);
        let mut next = mat![vec![]; input.n; input.n];
        for s in list.into_iter().flatten() {
            next[s.ps[0]][s.ps[1]].push(s);
        }
        beam = vec![];
        for i in 0..input.n {
            for j in 0..input.n {
                next[i][j].sort_by_key(|s| s.diff);
                next[i][j].truncate(300);
                for s in next[i][j].drain(..) {
                    beam.push(s);
                }
            }
        }
    }
    return vec![];
}

struct Set {
    list: Vec<Mutex<Vec<u32>>>,
}

impl Set {
    fn new() -> Self {
        let mut list = Vec::with_capacity(1 << 30);
        for _ in 0..1usize << 30 {
            list.push(Mutex::new(vec![]));
        }
        Self { list }
    }
    fn insert(&self, v: u64) -> bool {
        let h = (v as usize) & (self.list.len() - 1);
        let v = (v >> 32) as u32;
        let mut list = self.list[h].lock().unwrap();
        if !list.contains(&v) {
            list.push(v);
            true
        } else {
            false
        }
    }
}

pub fn greedy(input: &Input) -> Vec<usize> {
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(48392080);
    let mut min = 1000;
    let mut best = vec![];
    while get_time() < 60.0 {
        let moves = *[[0, 1], [0, 3], [2, 1], [2, 3]].choose(&mut rng).unwrap();
        let mut crt = input.start.clone();
        let mut out = vec![];
        for _ in 0..min {
            if diff(&crt, &input.target) <= input.wildcard {
                min = out.len();
                best = out;
                eprintln!("{:.3}: {}", get_time(), min);
                break;
            }
            let d0 = diff(&apply(&crt, &input.moves[moves[0]]), &input.target);
            let d1 = diff(&apply(&crt, &input.moves[moves[1]]), &input.target);
            if d0 < d1 || d0 == d1 && rng.gen_bool(0.5) {
                crt = apply(&crt, &input.moves[moves[0]]);
                out.push(moves[0]);
            } else {
                crt = apply(&crt, &input.moves[moves[1]]);
                out.push(moves[1]);
            }
        }
    }
    best
}

pub struct Trace {
    log: Vec<usize>,
}

impl Trace {
    pub fn new() -> Self {
        Trace { log: vec![] }
    }
    pub fn add(&mut self, c: usize, p: usize) -> usize {
        self.log.push(c << 60 | p & ((1usize << 60) - 1));
        self.log.len() - 1
    }
    pub fn get(&self, mut i: usize) -> Vec<usize> {
        let mut out = vec![];
        while i != ((1usize << 60) - 1) {
            out.push(self.log[i] >> 60);
            i = self.log[i] & ((1usize << 60) - 1);
        }
        out.reverse();
        out
    }
}
