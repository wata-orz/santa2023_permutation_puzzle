#![allow(non_snake_case)]

use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use solution::*;
use std::collections::HashMap;
use tools::Input;

fn main() {
    get_time();
    let input = read_input();
    if !input.puzzle_type.starts_with("cube") {
        return;
    }
    edge(&input);
}

pub fn edge(input: &Input) {
    let n = f64::sqrt((input.n / 6) as f64).round() as usize;
    eprintln!("n = {}", n);
    assert!(n <= 255);
    let x = 0;
    let y = 1;
    let mut moves = vec![];
    let list = vec![
        1, 7, 8, 14, 17, 23, 24, 30, 33, 39, 40, 46, 49, 55, 56, 62, 65, 71, 72, 78, 81, 87, 88, 94,
    ];
    let list2 = vec![
        50, 34, 66, 18, 13, 36, 75, 82, 11, 52, 27, 91, 2, 68, 43, 93, 4, 20, 59, 84, 29, 45, 77, 61,
    ];
    // let list = vec![
    //     1, 3, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34, 37, 39, 41, 43, 46, 48, 50, 52,
    // ];
    // let list2 = vec![
    //     28, 37, 19, 10, 7, 41, 21, 46, 5, 14, 30, 50, 1, 23, 39, 52, 3, 32, 12, 48, 16, 43, 25, 34,
    // ];
    for i in 0..6 {
        moves.push(i * n);
        moves.push(i * n + n - 1);
        moves.push(i * n + x);
        moves.push(i * n + n - 1 - x);
        moves.push(i * n + y);
        moves.push(i * n + n - 1 - y);
    }
    moves.sort();
    moves.dedup();
    eprintln!("#moves = {}", moves.len());
    eprintln!("cluster = {}", list.len());

    let mut mins = mat![vec![]; list.len(); list.len()];
    'lp: for depth in 4..=6 {
        eprintln!("depth = {}", depth);
        let mut visited = HashMap::new();
        let mut trace = Trace::new();
        rec(
            &(0..input.n as u8).collect_vec(),
            !0,
            &moves,
            depth,
            &mut visited,
            &mut trace,
            input,
        );
        eprintln!("#perms = {}", visited.len());
        for i2 in 1..list.len() {
            for i3 in i2 + 1..list.len() {
                if mins[i2][i3].len() > 0 {
                    continue;
                }
                let mut target = (0..input.n as u8).collect_vec();
                target[list[0]] = list[i3] as u8;
                target[list[i2]] = list[0] as u8;
                target[list[i3]] = list[i2] as u8;
                target[list2[0]] = list2[i3] as u8;
                target[list2[i2]] = list2[0] as u8;
                target[list2[i3]] = list2[i2] as u8;
                let tmp = visited
                    .par_iter()
                    .filter(|(perm, _)| visited.contains_key(&inv(&apply(perm, &target))))
                    .collect::<Vec<_>>();
                for (perm, id) in tmp {
                    let id2 = &visited[&inv(&apply(&perm, &target))];
                    for id in id {
                        for id2 in id2 {
                            let mut mv = trace.get(*id2);
                            mv.extend(trace.get(*id));
                            mins[i2][i3].push(mv);
                        }
                    }
                }
            }
        }
        let mut count = 0;
        for i2 in 1..list.len() {
            for i3 in i2 + 1..list.len() {
                if mins[i2][i3].len() > 0 {
                    count += 1;
                }
            }
        }
        eprintln!("count = {}", count);
        if count == 23 * 22 / 2 {
            break 'lp;
        }
    }
    for i2 in 1..list.len() {
        for i3 in i2 + 1..list.len() {
            if !mins[i2][i3].is_empty() {
                eprintln!(
                    "{} {} {} {}",
                    0,
                    i2,
                    i3,
                    mins[i2][i3]
                        .iter()
                        .map(|mv| mv.iter().map(|&op| input.move_names[op].clone()).join("."))
                        .join(" ")
                );
            }
        }
    }
}

pub fn inner(input: &Input) {
    let n = f64::sqrt((input.n / 6) as f64).round() as usize;
    eprintln!("n = {}", n);
    assert!(n <= 255);
    let x = 1;
    let y = 2;
    let mut moves = vec![];
    let mut list = vec![];
    for i in 0..6 {
        moves.push(i * n);
        moves.push(i * n + n - 1);
        moves.push(i * n + x);
        moves.push(i * n + n - 1 - x);
        moves.push(i * n + y);
        moves.push(i * n + n - 1 - y);
        list.push(i * n * n + x * n + y);
        list.push(i * n * n + (n - 1 - y) * n + x);
        list.push(i * n * n + (n - 1 - x) * n + (n - 1 - y));
        list.push(i * n * n + y * n + (n - 1 - x));
    }
    moves.sort();
    moves.dedup();
    list.sort();
    list.dedup();
    eprintln!("#moves = {}", moves.len());
    eprintln!("cluster = {}", list.len());

    let mut mins = mat![vec![]; list.len(); list.len()];
    'lp: for depth in 4..=5 {
        eprintln!("depth = {}", depth);
        let mut visited = HashMap::new();
        let mut trace = Trace::new();
        rec(
            &(0..input.n as u8).collect_vec(),
            !0,
            &moves,
            depth,
            &mut visited,
            &mut trace,
            input,
        );
        eprintln!("#perms = {}", visited.len());
        for i2 in 1..list.len() {
            for i3 in i2 + 1..list.len() {
                if mins[i2][i3].len() > 0 {
                    continue;
                }
                let mut target = (0..input.n as u8).collect_vec();
                target[list[0]] = list[i3] as u8;
                target[list[i2]] = list[0] as u8;
                target[list[i3]] = list[i2] as u8;
                let tmp = visited
                    .par_iter()
                    .filter(|(perm, _)| visited.contains_key(&inv(&apply(perm, &target))))
                    .collect::<Vec<_>>();
                for (perm, id) in tmp {
                    let id2 = &visited[&inv(&apply(&perm, &target))];
                    for id in id {
                        for id2 in id2 {
                            let mut mv = trace.get(*id2);
                            mv.extend(trace.get(*id));
                            mins[i2][i3].push(mv);
                        }
                    }
                }
            }
        }
        let mut count = 0;
        for i2 in 1..list.len() {
            for i3 in i2 + 1..list.len() {
                if mins[i2][i3].len() > 0 {
                    count += 1;
                }
            }
        }
        eprintln!("count = {}", count);
        if count == 23 * 22 / 2 {
            break 'lp;
        }
    }
    for i2 in 1..list.len() {
        for i3 in i2 + 1..list.len() {
            if !mins[i2][i3].is_empty() {
                eprintln!(
                    "{} {} {} {}",
                    0,
                    i2,
                    i3,
                    mins[i2][i3]
                        .iter()
                        .map(|mv| mv.iter().map(|&op| input.move_names[op].clone()).join("."))
                        .join(" ")
                );
            }
        }
    }
}

fn rec(
    crt: &Vec<u8>,
    id: u32,
    moves: &Vec<usize>,
    depth: usize,
    visited: &mut HashMap<Vec<u8>, Vec<u32>>,
    trace: &mut Trace<usize>,
    input: &Input,
) {
    if depth == 0 {
        visited.entry(crt.clone()).or_insert(vec![]).push(id);
        return;
    }
    for &mv in moves {
        let next = apply(crt, &input.moves[mv]);
        let id = trace.add(mv, id);
        rec(&next, id, moves, depth - 1, visited, trace, input);
    }
}

pub fn inv(perm: &[u8]) -> Vec<u8> {
    let mut inv = vec![0; perm.len()];
    for i in 0..perm.len() {
        inv[perm[i] as usize] = i as u8;
    }
    inv
}
