#![allow(non_snake_case)]

// 大きいサイズ用のglobe solver

use itertools::Itertools;
use rand::prelude::*;
use rayon::prelude::*;
use solution::*;
use std::{
    collections::{HashSet, VecDeque},
    iter::repeat,
};
use tools::Input;

fn main() {
    get_time();
    let input0 = read_input();
    if !input0.puzzle_type.starts_with("globe") {
        return;
    }
    // if input0.id < 388 {
    //     return;
    // }
    let input = Globe::new(&input0);
    let start = input0.start.clone();
    let Globe { rows, cols, .. } = input;
    dbg!(rows, cols, input.w, input.wildcard);
    let out = if input.rows % 2 != 0 {
        let mut out = vec![];
        // 中央は反転されないので、左右にローテートするだけで完成
        let r = rows / 2;
        let mut rot = !0;
        for i in 0..cols {
            if (0..cols).all(|j| start[r * cols + (j + i) % cols] == input.target[r * cols + j]) {
                if rot == !0 || rot.min(cols - rot) > i.min(cols - i) {
                    rot = i;
                }
            }
        }
        assert!(rot != !0);
        if rot < cols - rot {
            out.extend(repeat(cols + r).take(rot));
        } else {
            out.extend(repeat(cols + r + input.m).take(cols - rot));
        }
        let rs = (0..r).collect_vec();
        let (input2, start2) = input.subinput(&rs, &start);
        let out1 = solve(&input2, &start2);
        out.extend(input.convert_from_suboutput(&rs, &out1));
        out
    } else {
        solve(&input, &start)
    };
    write_output(&input0, &optimize_bisearch(&input0, &out));
    write_profile();
}

fn solve(input: &Globe, start: &Vec<u16>) -> Vec<usize> {
    let mut out = init_optimize(input, &mut start.clone(), if input.rows == 8 { 5.0 } else { 4.0 });
    profile!(insert);
    let n2 = 2 * input.cols;
    let m2 = 2 + input.cols;
    let (to, trace3, dist3, rot3) = {
        let input = input.subinput(&[0], &start).0;
        let mut to = mat![!0; m2 * 2; n2];
        for i in 0..m2 * 2 {
            if i >= m2 && i < m2 + input.cols {
                continue;
            }
            let mut start = (0..n2).collect_vec();
            input.apply(&mut start, i);
            for j in 0..n2 {
                to[i][start[j]] = j;
            }
        }
        let mut trace3 = Trace::new();
        let mut dist3 = mat![(1000000000, !0); n2; n2; n2];
        let mut rs = vec![input.cols, input.cols + 1, input.cols + m2, input.cols + 1 + m2];
        // rot3[i][j][k] := i <- j <- k <- i となるような操作の列
        let mut rot3 = mat![vec![]; n2; n2; n2];
        for c in 0..input.cols {
            let mut moves = vec![];
            loop {
                moves.push([c, rs[0], c, rs[1], c, rs[2], c, rs[3]]);
                moves.push([rs[0], c, rs[1], c, rs[2], c, rs[3], c]);
                if !next_permutation(&mut rs) {
                    break;
                }
            }
            for mv in moves {
                let mut crt = (0..input.n).collect_vec();
                for &op in &mv {
                    input.apply(&mut crt, op);
                }
                let loops = get_loops(&crt);
                if loops.len() == 1 && loops[0].len() == 3 {
                    for k in 0..3 {
                        rot3[loops[0][k]][loops[0][(k + 1) % 3]][loops[0][(k + 2) % 3]].push(mv);
                    }
                }
            }
        }
        let mut que = VecDeque::new();
        for i in 0..n2 {
            for j in 0..n2 {
                for k in 0..n2 {
                    if rot3[i][j][k].len() > 0 {
                        dist3[i][j][k] = (0, trace3.add(i * n2 * n2 + j * n2 + k, !0));
                        que.push_back((i, j, k));
                    }
                }
            }
        }
        while let Some((i, j, k)) = que.pop_front() {
            let (d, id) = dist3[i][j][k];
            for op in 0..m2 * 2 {
                if op >= m2 && op < m2 + input.cols {
                    continue;
                }
                let i2 = to[op][i];
                let j2 = to[op][j];
                let k2 = to[op][k];
                if dist3[i2][j2][k2].0.setmin(d + 1) {
                    dist3[i2][j2][k2].1 = trace3.add(op, id);
                    que.push_back((i2, j2, k2));
                }
            }
        }
        (to, trace3, dist3, rot3)
    };
    const B: usize = 10000;
    let mut beam = vec![vec![]; 8];
    let mut cand = vec![vec![]; 8];
    let mut crt = start.clone();
    for &op in &out {
        input.apply(&mut crt, op);
    }
    let init_diff = diff(&crt, &input.target);
    beam[0].push((crt, out));
    let mut min_len = 1000000000;
    out = vec![];
    for g in 0..=init_diff {
        if init_diff - g < input.wildcard.max(3) - 3 {
            break;
        }
        if beam[g & 7].len() > 0 {
            eprintln!(
                "{:.3}: diff = {}, len = {} - {} (# = {})",
                get_time(),
                init_diff - g,
                beam[g & 7][0].1.len(),
                beam[g & 7].last().unwrap().1.len(),
                beam[g & 7].len()
            );
            vis(&input, &beam[g & 7][0].0);
            if init_diff - g <= input.wildcard && min_len.setmin(beam[g & 7][0].1.len()) {
                out = beam[g & 7][0].1.clone();
            }
            let next = (0..beam[g & 7].len())
                .into_par_iter()
                .map(|b| {
                    let mut next = vec![];
                    let (crt, out) = &beam[g & 7][b];
                    let mut list = vec![];
                    for r in 0..input.rows / 2 {
                        let ps = (r * input.cols..(r + 1) * input.cols)
                            .chain((input.rows - r - 1) * input.cols..(input.rows - r) * input.cols)
                            .collect_vec();
                        for i in 0..n2 {
                            for j in i + 1..n2 {
                                for k in i + 1..n2 {
                                    if j == k {
                                        continue;
                                    }
                                    let mut gain = 0i32;
                                    if crt[ps[i]] == input.target[ps[i]] {
                                        gain -= 1;
                                    }
                                    if crt[ps[j]] == input.target[ps[i]] {
                                        gain += 1;
                                    }
                                    if crt[ps[j]] == input.target[ps[j]] {
                                        gain -= 1;
                                    }
                                    if crt[ps[k]] == input.target[ps[j]] {
                                        gain += 1;
                                    }
                                    if crt[ps[k]] == input.target[ps[k]] {
                                        gain -= 1;
                                    }
                                    if crt[ps[i]] == input.target[ps[k]] {
                                        gain += 1;
                                    }
                                    if gain > 0 {
                                        list.push((r, ps[i], ps[j], ps[k], gain));
                                    }
                                }
                            }
                        }
                    }
                    let max_gain = list.iter().map(|&(_, _, _, _, gain)| gain).max().unwrap_or(0);
                    if max_gain >= 2 {
                        list.retain(|&(_, _, _, _, gain)| gain >= 2);
                    }
                    let mut idss = vec![vec![]; out.len() + 1];
                    {
                        let mut ps = (0..input.n).collect_vec();
                        let mut ids = vec![!0; input.n];
                        for t in (0..=out.len()).rev() {
                            for r in 0..input.rows / 2 {
                                for i in 0..input.cols {
                                    ids[ps[r * input.cols + i]] = i;
                                }
                            }
                            for r in 0..input.rows / 2 {
                                for i in 0..input.cols {
                                    ids[ps[(input.rows - r - 1) * input.cols + i]] = input.cols + i;
                                }
                            }
                            idss[t] = ids.clone();
                            if t > 0 {
                                input.apply(&mut ps, input.rev(out[t - 1]));
                            }
                        }
                    }
                    for (r, i, j, k, gain) in list {
                        let mut min_cost = 1000000000;
                        let mut min = (0, vec![]);
                        for t in 0..=out.len() {
                            let ids = &idss[t];
                            let i = ids[i];
                            let j = ids[j];
                            let k = ids[k];
                            if dist3[i][j][k].0 == 0 {
                                for mv in &rot3[i][j][k] {
                                    let add = mv
                                        .iter()
                                        .map(|&op| {
                                            if op < input.cols {
                                                op
                                            } else if op == input.cols {
                                                input.cols + r
                                            } else if op == input.cols + 1 {
                                                input.cols + input.rows - r - 1
                                            } else if op == m2 + input.cols {
                                                input.cols + r + input.m
                                            } else if op == m2 + input.cols + 1 {
                                                input.cols + input.rows - r - 1 + input.m
                                            } else {
                                                panic!()
                                            }
                                        })
                                        .collect_vec();
                                    let ov = overlap(input, &out[..t], &add) + overlap(input, &add, &out[t..]);
                                    let cost = add.len() as i32 - ov;
                                    if min_cost.setmin(cost) {
                                        min = (t, add);
                                    }
                                }
                            } else {
                                let mut ss = vec![!0];
                                if t > 0 {
                                    ss.push(input.rev(out[t - 1]));
                                    for t2 in (1..t).rev() {
                                        if out[t2] < input.cols || out[t2 - 1] < input.cols {
                                            break;
                                        }
                                        ss.push(input.rev(out[t2 - 1]));
                                    }
                                }
                                if t < out.len() {
                                    ss.push(out[t]);
                                    for t2 in t + 1..out.len() {
                                        if out[t2] < input.cols || out[t2 - 1] < input.cols {
                                            break;
                                        }
                                        ss.push(out[t2]);
                                    }
                                }
                                ss.sort();
                                ss.dedup();
                                for s in ss {
                                    let s = if s == !0 || s < input.cols {
                                        s
                                    } else if s == input.cols + r {
                                        input.cols
                                    } else if s == input.cols + input.rows - r - 1 {
                                        input.cols + 1
                                    } else if s == input.cols + r + input.m {
                                        input.cols + m2
                                    } else if s == input.cols + input.rows - r - 1 + input.m {
                                        input.cols + 1 + m2
                                    } else {
                                        continue;
                                    };
                                    let (i, j, k) = if s == !0 { (i, j, k) } else { (to[s][i], to[s][j], to[s][k]) };
                                    let mut mv = trace3.get(dist3[i][j][k].1);
                                    let r3 = mv.remove(0);
                                    if s < input.cols {
                                        mv.push(s);
                                    } else if s < m2 {
                                        mv.push(s + m2);
                                    } else if s < m2 * 2 {
                                        mv.push(s - m2);
                                    }
                                    let add = mv
                                        .iter()
                                        .rev()
                                        .map(|&op| if op < input.cols { op } else { (op + m2) % (2 * m2) })
                                        .chain(rot3[r3 / n2 / n2][r3 / n2 % n2][r3 % n2][0].iter().cloned())
                                        .chain(mv.iter().cloned())
                                        .map(|op| {
                                            if op < input.cols {
                                                op
                                            } else if op == input.cols {
                                                input.cols + r
                                            } else if op == input.cols + 1 {
                                                input.cols + input.rows - r - 1
                                            } else if op == m2 + input.cols {
                                                input.cols + r + input.m
                                            } else if op == m2 + input.cols + 1 {
                                                input.cols + input.rows - r - 1 + input.m
                                            } else {
                                                panic!()
                                            }
                                        })
                                        .collect_vec();
                                    let ov = overlap(input, &out[..t], &add) + overlap(input, &add, &out[t..]);
                                    let cost = add.len() as i32 - ov;
                                    if min_cost.setmin(cost) {
                                        min = (t, add);
                                    }
                                }
                            }
                        }
                        assert!(min_cost < 1000000000);
                        let (t, add) = min;
                        next.push((
                            gain,
                            (out.len() as i32 + min_cost, thread_rng().gen::<u32>()),
                            g,
                            b,
                            t,
                            add,
                            (i, j, k),
                        ))
                    }
                    next
                })
                .collect::<Vec<_>>();
            for (gain, len, g, b, t, add, rot) in next.into_iter().flatten() {
                cand[(g + gain as usize) & 7].push((len, g, b, t, add, rot));
            }
        } else {
            eprintln!("{:.3}: diff = {}", get_time(), init_diff - g,);
        }
        beam[(g + 1) & 7].clear();
        eprintln!("#cand = {}", cand[(g + 1) & 7].len());
        cand[(g + 1) & 7].par_sort_by_key(|a| a.0);
        let mut used = HashSet::new();
        for &(_, g2, b, t, ref add, rot) in &cand[(g + 1) & 7] {
            let (crt, out) = &beam[g2 & 7][b];
            let mut crt = crt.clone();
            rotate(&mut crt, &[rot.0, rot.1, rot.2]);
            if used.insert(crt.clone()) {
                let mut out2 = out[..t].to_vec();
                extend_with_overlap(input, &mut out2, add);
                extend_with_overlap(input, &mut out2, &out[t..]);
                beam[(g + 1) & 7].push((crt, out2));
                if beam[(g + 1) & 7].len() >= B {
                    break;
                }
            }
        }
        cand[(g + 1) & 7].clear();
    }
    out
}

fn init_optimize(input: &Globe, start: &mut Vec<u16>, avg: f64) -> Vec<usize> {
    assert!(input.rows % 2 == 0);
    let mut rng = rand_pcg::Pcg64Mcg::new(8901234);
    let mut hash = mat![0; input.n; input.colors];
    for i in 0..input.n {
        for j in 0..input.colors {
            hash[i][j] = rng.gen::<u64>();
        }
    }
    let mut outs = vec![vec![]; input.rows / 2 + 1];
    {
        profile!(climb);
        let start0 = start.clone();
        let f_odd = if (input.cols / 2) % 2 != 0 {
            (1 << (input.rows / 2)) - 1
        } else {
            0
        };
        let mut odd = if input.w == 1 {
            let mut to = vec![!0; input.colors];
            for i in 0..input.n {
                to[input.target[i] as usize] = i;
            }
            let mut used = vec![false; input.n];
            let mut odd = 0;
            for r0 in 0..input.rows / 2 {
                for r in [r0, input.rows - 1 - r0] {
                    for c in 0..input.cols {
                        if !used[r * input.cols + c] {
                            let mut p = r * input.cols + c;
                            let mut len = 0;
                            loop {
                                if !used[p].setmax(true) {
                                    break;
                                }
                                p = to[start[p] as usize];
                                len += 1;
                            }
                            if len % 2 == 0 {
                                odd ^= 1 << r0;
                            }
                        }
                    }
                }
            }
            odd
        } else {
            0
        };
        let avg = 1.0;
        let mut out = vec![];
        macro_rules! evaluate {
            () => {{
                let mut start = start0.clone();
                for &op in &out {
                    input.apply(&mut start, op);
                }
                (out.len() as f64 + State::new(input, start, odd, !0).eval() as f64 * avg) / input.n as f64
            }};
        }
        macro_rules! change_odd {
            ($op:expr) => {{
                let op = $op;
                if op < input.cols {
                    odd ^= f_odd;
                } else {
                    let r = (op - input.cols) % input.m;
                    let r = r.min(input.rows - r - 1);
                    odd ^= 1 << r;
                }
            }};
        }
        let mut best = 1e10;
        let mut best_out = out.clone();
        let max_len = 40;
        const TL: usize = 100000;
        for iter in 0..TL {
            let kick = 3;
            out = best_out.clone();
            let mut crt = evaluate!();
            for t in 0..1000 {
                let ty = rng.gen_range(0..4);
                if ty == 0 {
                    if out.len() <= 1 {
                        continue;
                    }
                    let (i, j) = if rng.gen_bool(0.5) {
                        let i = rng.gen_range(0..out.len() - 1);
                        (i, i + 1)
                    } else {
                        let i = rng.gen_range(0..out.len() - 1);
                        let mut j = rng.gen_range(0..out.len() - 1);
                        if j <= i {
                            j += 1;
                        }
                        (i, j)
                    };
                    out.swap(i, j);
                    let next = evaluate!();
                    if t < kick || crt >= next {
                        crt = next;
                    } else {
                        out.swap(i, j);
                    }
                } else if ty == 1 {
                    if out.len() == 0 {
                        continue;
                    }
                    let i = rng.gen_range(0..out.len());
                    let mut op = if rng.gen_bool(0.5) {
                        rng.gen_range(0..input.cols)
                    } else {
                        rng.gen_range(0..input.rows) + input.cols + rng.gen_range(0..2) * input.m
                    };
                    change_odd!(out[i]);
                    std::mem::swap(&mut op, &mut out[i]);
                    change_odd!(out[i]);
                    let next = evaluate!();
                    if t < kick || crt >= next {
                        crt = next;
                    } else {
                        change_odd!(out[i]);
                        std::mem::swap(&mut op, &mut out[i]);
                        change_odd!(out[i]);
                    }
                } else if ty == 2 {
                    if out.len() + 1 >= max_len {
                        continue;
                    }
                    let i = rng.gen_range(0..=out.len());
                    let op = if rng.gen_bool(0.5) {
                        rng.gen_range(0..input.cols)
                    } else {
                        rng.gen_range(0..input.rows) + input.cols + rng.gen_range(0..2) * input.m
                    };
                    change_odd!(op);
                    out.insert(i, op);
                    let next = evaluate!();
                    if t < kick || crt >= next {
                        crt = next;
                    } else {
                        change_odd!(op);
                        out.remove(i);
                    }
                } else {
                    if out.len() == 0 {
                        continue;
                    }
                    let i = rng.gen_range(0..out.len());
                    let op = out[i];
                    change_odd!(op);
                    out.remove(i);
                    let next = evaluate!();
                    if t < kick || crt >= next {
                        crt = next;
                    } else {
                        change_odd!(op);
                        out.insert(i, op);
                    }
                }
            }
            if best.setmin(crt) {
                best_out = out;
                eprintln!("{:.3}: {:.5} (len = {})", iter as f64 / 100000.0, best, best_out.len());
            }
        }
        out = best_out;
        for &op in &out {
            input.apply(start, op);
        }
        outs[0] = out;
    }
    let input0 = input;
    let mut rots = vec![!0; input.rows];
    'lp: for r0 in 0..input.rows / 2 {
        eprintln!("###############################################");
        let (input, start1) = input.subinput(&[r0], start);
        let Globe { rows, cols, n, m, .. } = input;
        let mut basic_moves = vec![];
        for r in 0..rows {
            basic_moves.push((r.min(rows - r - 1), vec![cols + r]));
        }
        for j in 0..cols {
            for r in 0..rows / 2 {
                basic_moves.push((r, vec![j, cols + r, j]));
                basic_moves.push((r, vec![j, cols + r + m, j]));
                basic_moves.push((r, vec![j, cols + r, j, cols + rows - r - 1, j, cols + rows - r - 1 + m, j]));
                basic_moves.push((
                    r,
                    vec![j, cols + rows - r - 1, j, cols + rows - r - 1 + m, j, cols + r + m, j],
                ));
                basic_moves.push((r, vec![j, cols + rows - r - 1, j, cols + r, j, cols + r + m, j]));
                basic_moves.push((r, vec![j, cols + r, j, cols + r + m, j, cols + rows - r - 1 + m, j]));
            }
        }
        let basic_moves = basic_moves
            .into_iter()
            .map(|(r, moves)| {
                let mut crt = (0..n).collect_vec();
                for &op in &moves {
                    input.apply(&mut crt, op);
                }
                let loops = get_loops(&crt);
                assert_eq!(loops.len(), 1);
                (r, moves, loops[0].clone())
            })
            .collect_vec();
        let mut trace = Trace::new();
        let mut beam = vec![vec![]; 16];
        let mut cand = vec![vec![]; 16];
        let init_odd = if input.w == 1 {
            let mut to = vec![!0; input.colors];
            for i in 0..input.n {
                to[input.target[i] as usize] = i;
            }
            let mut used = vec![false; input.n];
            let mut odd = 0;
            for r0 in 0..rows / 2 {
                for r in [r0, rows - 1 - r0] {
                    for c in 0..cols {
                        if !used[r * cols + c] {
                            let mut p = r * cols + c;
                            let mut len = 0;
                            loop {
                                if !used[p].setmax(true) {
                                    break;
                                }
                                p = to[start1[p] as usize];
                                len += 1;
                            }
                            if len % 2 == 0 {
                                odd ^= 1 << r0;
                            }
                        }
                    }
                }
            }
            odd
        } else {
            0
        };
        beam[0].push(State::new(&input, start1.clone(), init_odd, !0));
        let mut used = HashSet::new();
        used.insert(get_hash(&start1, &hash, rows, cols, true));
        const B: usize = 1000000;
        const INC: i32 = 0;
        let max_size = (B / cols).max(1);
        dbg!(max_size);
        let wildcard = (input0.wildcard * (r0 + 1) / (input0.rows / 2) - input0.wildcard * r0 / (input0.rows / 2)) as i32;
        vis(&input, &start1);
        let mut best = BoundedSortedList::new(100);
        // let mut best = BoundedSortedList::new(1000);
        let mut min = 1e10;
        for t in 0.. {
            if t as f64 >= min {
                let mut out = vec![];
                let mut min_rot = (0, 0);
                let mut min = 1e10;
                for (_, (id, rot_high, rot_low, diff)) in best.list() {
                    rots[r0] = rot_high as usize;
                    rots[input0.rows - r0 - 1] = rot_low as usize;
                    let mut add =
                        input0.convert_from_suboutput(&[r0], &trace.get(id).into_iter().map(|op| op as usize).collect_vec());
                    shrink(&input0, &mut add);
                    outs[1 + r0] = add;
                    let len = optimize(&input0, &outs, &rots).len();
                    if min.setmin(len as f64 + diff as f64 * avg) {
                        eprintln!("{} (len = {}, diff = {})", len as f64 + diff as f64 * avg, len, diff);
                        out = outs[1 + r0].clone();
                        min_rot = (rot_high, rot_low);
                    }
                }
                for &op in &out {
                    input0.apply(start, op);
                }
                rots[r0] = min_rot.0 as usize;
                rots[input0.rows - r0 - 1] = min_rot.1 as usize;
                outs[1 + r0] = out;
                continue 'lp;
            }
            if beam[t & 15].len() > 0 {
                eprintln!(
                    "{:.3}: t = {}, eval = {} - {}",
                    get_time(),
                    t,
                    beam[t & 15][0].eval(),
                    beam[t & 15].last().unwrap().eval()
                );
                vis(&input, &beam[t & 15][0].crt);
                for b in 0..beam[t & 15].len() {
                    let diff = (beam[t & 15][b].diff[0] as i32 - wildcard).max(0);
                    min.setmin(t as f64 + diff as f64 * avg);
                    if best.can_insert(t as f64 + diff as f64 * avg) {
                        let (_, _, _, rot_high, rot_low) = rotate_diff1(&input, &beam[t & 15][b].crt, 0, beam[t & 15][b].odd);
                        best.insert(t as f64 + diff as f64 * avg, (beam[t & 15][b].id, rot_high, rot_low, diff));
                    }
                }
                profile!(par_list);
                let list = (0..beam[t & 15].len())
                    .into_par_iter()
                    .map(|b| {
                        let mut list = vec![];
                        let mut crt = beam[t & 15][b].crt.clone();
                        let eval = beam[t & 15][b].eval();
                        let prev = if beam[t & 15][b].id == !0 {
                            !0
                        } else {
                            let prev = trace.prev_move(beam[t & 15][b].id) as usize;
                            if prev >= cols {
                                (prev + m) % (2 * m)
                            } else {
                                prev
                            }
                        } as usize;
                        for (k, &(r, ref moves, ref ps)) in basic_moves.iter().enumerate() {
                            let ov = if prev == moves[0] { 2 } else { 0 };
                            rotate(&mut crt, ps);
                            let adj_diff = adj_diff1(&input, &crt, r);
                            let odd = beam[t & 15][b].odd ^ (1 << r);
                            let (diff, diff2, _, _, _) = rotate_diff1(&input, &crt, r, odd);
                            let next_eval = eval
                                - compute_eval(beam[t & 15][b].adj_diff[r], beam[t & 15][b].diff[r], beam[t & 15][b].diff2[r])
                                + compute_eval(adj_diff, diff, diff2);
                            if next_eval <= eval + INC {
                                list.push((
                                    (moves.len() - ov) as u8,
                                    (
                                        (next_eval as i16, thread_rng().gen::<u16>()),
                                        (t as u8) & 15,
                                        b as u32,
                                        (cols + k) as u16,
                                    ),
                                ));
                            }
                            rotate_rev(&mut crt, ps);
                        }
                        list
                    })
                    .collect::<Vec<_>>();
                for (k, entry) in list.into_iter().flatten() {
                    cand[(t + k as usize) & 15].push(entry);
                }
            } else {
                eprintln!("{}:", t);
            }
            beam[(t + 1) & 15].clear();
            eprintln!("#cand = {}", cand[(t + 1) & 15].len());
            {
                profile!(sort);
                cand[(t + 1) & 15].par_sort_by_key(|a| a.0);
            }
            let list = {
                profile!(par_next);
                let list = cand[(t + 1) & 15]
                    .par_iter()
                    .map(|&(_, t2, b, k)| {
                        let mut next = beam[t2 as usize][b as usize].crt.clone();
                        rotate(&mut next, &basic_moves[k as usize - cols].2);
                        let r = basic_moves[k as usize - cols].0;
                        let mut odd = beam[t2 as usize][b as usize].odd;
                        odd ^= 1 << r;
                        let (diff, diff2, rot) = {
                            let mut rot = beam[t2 as usize][b as usize].rotate.clone();
                            let (d, d2, rr, _, _) = rotate_diff1(&input, &next, r, odd);
                            rot[r] = rr;
                            let mut diff = beam[t2 as usize][b as usize].diff.clone();
                            let mut diff2 = beam[t2 as usize][b as usize].diff2.clone();
                            let mut rot = beam[t2 as usize][b as usize].rotate.clone();
                            diff[r] = d;
                            diff2[r] = d2;
                            rot[r] = rr;
                            (diff, diff2, rot)
                        };
                        (t2, b, k, get_hash(&next, &hash, rows, cols, true), diff, diff2, rot, odd)
                    })
                    .collect::<Vec<_>>();
                list
            };
            profile!(single_next);
            let mut open = (rows / 2) * cols;
            let mut count = mat![0; rows / 2; cols];
            for (t2, b, k, hash, diff, diff2, rot, odd) in list {
                if (0..rows / 2).any(|r| count[r][rot[r] as usize] < max_size) && used.insert(hash) {
                    let mut next = beam[t2 as usize][b as usize].crt.clone();
                    if (k as usize) < cols {
                        input.apply(&mut next, k as usize);
                    } else {
                        rotate(&mut next, &basic_moves[k as usize - cols].2);
                    }
                    let k = k as usize;
                    let t2 = t2 as usize;
                    let b = b as usize;
                    for r in 0..rows / 2 {
                        count[r][rot[r] as usize] += 1;
                        if count[r][rot[r] as usize] == max_size {
                            open -= 1;
                        }
                    }
                    let adj_diff = if k < cols {
                        let mut adj_diff = [0; 4];
                        for r in 0..rows / 2 {
                            adj_diff[r] = adj_diff1(&input, &next, r);
                        }
                        adj_diff
                    } else {
                        let r = basic_moves[k - cols].0;
                        let mut adj_diff = beam[t2][b].adj_diff.clone();
                        adj_diff[r] = adj_diff1(&input, &next, r);
                        adj_diff
                    };
                    let id = if k < cols {
                        trace.add(k as u32, beam[t2][b].id)
                    } else {
                        basic_moves[k - cols]
                            .1
                            .iter()
                            .fold(beam[t2][b].id, |id, &op| trace.add(op as u32, id))
                    };
                    beam[(t + 1) & 15].push(State {
                        crt: next,
                        adj_diff,
                        diff,
                        diff2,
                        rotate: rot,
                        odd,
                        id,
                    });
                    if open == 0 {
                        break;
                    }
                }
            }
            cand[(t + 1) & 15].clear();
        }
        write_profile();
        panic!();
    }
    eprint!("{} -> ", outs.iter().map(|a| a.len()).sum::<usize>());
    let mut out = optimize(&input, &outs, &rots);
    shrink(&input, &mut out);
    eprintln!("{}", out.len());
    out
}

#[derive(Clone, Debug)]
struct Globe {
    n: usize,
    m: usize,
    rows: usize,
    cols: usize,
    target: Vec<u16>,
    colors: usize,
    w: usize,
    wildcard: usize,
}

impl Globe {
    fn new(input: &Input) -> Self {
        Self {
            n: input.n,
            m: input.m,
            rows: input.move_names.iter().filter(|s| s.starts_with("r")).count() as usize,
            cols: input.move_names.iter().filter(|s| s.starts_with("f")).count() as usize,
            target: input.target.clone(),
            colors: *input.target.iter().max().unwrap() as usize + 1,
            w: input.target.iter().position(|&c| c != input.target[0]).unwrap(),
            wildcard: input.wildcard,
        }
    }
    fn subinput(&self, rs: &[usize], start: &[u16]) -> (Self, Vec<u16>) {
        let mut target = Vec::with_capacity(self.cols * rs.len() * 2);
        let mut start2 = Vec::with_capacity(self.cols * rs.len() * 2);
        for r in rs {
            target.extend_from_slice(&self.target[r * self.cols..(r + 1) * self.cols]);
            start2.extend_from_slice(&start[r * self.cols..(r + 1) * self.cols]);
        }
        for r in rs.iter().rev() {
            let r = self.rows - r - 1;
            target.extend_from_slice(&self.target[r * self.cols..(r + 1) * self.cols]);
            start2.extend_from_slice(&start[r * self.cols..(r + 1) * self.cols]);
        }
        (
            Self {
                n: rs.len() * 2 * self.cols,
                m: rs.len() * 2 + self.cols,
                rows: rs.len() * 2,
                cols: self.cols,
                target,
                ..*self
            },
            start2,
        )
    }
    fn convert_from_suboutput(&self, rs: &[usize], out: &[usize]) -> Vec<usize> {
        let rs = rs
            .iter()
            .cloned()
            .chain(rs.iter().rev().map(|&r| self.rows - r - 1))
            .collect_vec();
        out.iter()
            .map(|&op| {
                if op < self.cols {
                    op
                } else if op < self.cols + rs.len() {
                    self.cols + rs[op - self.cols]
                } else {
                    self.m + self.cols + rs[op - self.cols * 2 - rs.len()]
                }
            })
            .collect_vec()
    }
    fn apply<T: Copy>(&self, crt: &mut Vec<T>, op: usize) {
        let rows = self.rows;
        let cols = self.cols;
        if op < cols {
            let k = cols / 2;
            for r in 0..rows / 2 {
                let r2 = rows - r - 1;
                for i in 0..k {
                    crt.swap(r * cols + (op + i) % cols, r2 * cols + (op + k - i - 1) % cols);
                }
            }
        } else if op < self.m {
            let r = op - cols;
            crt[r * cols..(r + 1) * cols].rotate_left(1);
        } else {
            let r = op - cols - self.m;
            crt[r * cols..(r + 1) * cols].rotate_right(1);
        }
    }
    fn rev(&self, op: usize) -> usize {
        if op < self.cols {
            op
        } else {
            (op + self.m) % (self.m * 2)
        }
    }
}

fn rotate(crt: &mut Vec<u16>, ps: &[usize]) {
    let s = crt[ps[0]];
    for i in 0..ps.len() - 1 {
        crt[ps[i]] = crt[ps[i + 1]];
    }
    crt[ps[ps.len() - 1]] = s;
}

fn rotate_rev(crt: &mut Vec<u16>, ps: &[usize]) {
    let s = crt[ps[ps.len() - 1]];
    for i in (0..ps.len() - 1).rev() {
        crt[ps[i + 1]] = crt[ps[i]];
    }
    crt[ps[0]] = s;
}

#[derive(Clone, Debug)]
struct State {
    crt: Vec<u16>,
    adj_diff: [u8; 4],
    diff: [u8; 4],
    diff2: [u8; 4],
    rotate: [u8; 4],
    odd: u32,
    id: u32,
}

impl State {
    fn new(input: &Globe, crt: Vec<u16>, odd: u32, id: u32) -> Self {
        let adj_diff = adj_diff(input, &crt);
        let (diff, diff2, rotate) = rotate_diff(input, &crt, odd);
        Self {
            crt,
            adj_diff,
            diff,
            diff2,
            rotate,
            odd,
            id,
        }
    }
    fn eval(&self) -> i32 {
        (0..self.adj_diff.len())
            .map(|r| compute_eval(self.adj_diff[r], self.diff[r], self.diff2[r]))
            .sum()
    }
}

fn compute_eval(adj_diff: u8, diff: u8, diff2: u8) -> i32 {
    adj_diff as i32 + 2 * diff as i32 + diff2 as i32
}

fn get_hash(crt: &Vec<u16>, hash: &Vec<Vec<u64>>, rows: usize, cols: usize, normalize: bool) -> u64 {
    let mut h = 0;
    let mut rs = [0; 4];
    if normalize {
        for i in 0..rows / 2 {
            rs[i] = (0..cols).min_by_key(|&j| crt[i * cols + j]).unwrap();
        }
    }
    for r in 0..rows {
        let mut s = rs[r.min(rows - r - 1)];
        for i in 0..cols {
            h ^= hash[r * cols + i][crt[r * cols + s] as usize];
            s += 1;
            if s >= cols {
                s -= cols;
            }
        }
    }
    h
}

const W0: i32 = 1;
const W1: i32 = 1;

fn adj_diff1(input: &Globe, crt: &Vec<u16>, r: usize) -> u8 {
    let Globe {
        rows, cols, ref target, ..
    } = *input;
    let mut d = 2 * input.cols as i32 * W1;
    let mut used0 = 0u128;
    let mut used1 = 0u128;
    let c0 = target[r * cols];
    let d0 = target[r * cols + cols - 1];
    let c1 = target[(rows - r - 1) * cols];
    let d1 = target[(rows - r - 1) * cols + cols - 1];
    for i in 0..cols {
        let p = crt[r * cols + i];
        let q = crt[r * cols + (i + 1) % cols];
        if p < c1 {
            if p == q {
                d -= W1;
            } else if q < c1 && (p + 1 == q || p == d0 && q == c0) {
                if used0 >> (p - c0) & 1 == 0 {
                    used0 |= 1 << (p - c0);
                    d -= W1;
                }
            }
        } else {
            if p == q {
                d -= W0;
            } else if q >= c1 && (p - 1 == q || q == d1 && p == c1) {
                if used1 >> (q - c1) & 1 == 0 {
                    used1 |= 1 << (q - c1);
                    d -= W0;
                }
            }
        }
    }
    for i in 0..cols {
        let p = crt[(rows - r - 1) * cols + i];
        let q = crt[(rows - r - 1) * cols + (i + 1) % cols];
        if p < c1 {
            if p == q {
                d -= W0;
            } else if q < c1 && (p - 1 == q || q == d0 && p == c0) {
                if used0 >> (q - c0) & 1 == 0 {
                    used0 |= 1 << (q - c0);
                    d -= W0;
                }
            }
        } else {
            if p == q {
                d -= W1;
            } else if q >= c1 && (p + 1 == q || p == d1 && q == c1) {
                if used1 >> (p - c1) & 1 == 0 {
                    used1 |= 1 << (p - c1);
                    d -= W1;
                }
            }
        }
    }
    d as u8
}

fn adj_diff(input: &Globe, crt: &Vec<u16>) -> [u8; 4] {
    let mut ret = [0; 4];
    for r in 0..input.rows / 2 {
        ret[r] = adj_diff1(input, crt, r) as u8;
    }
    ret
}

/// (誤差, 正しい行にある間違った数, 上下のズレ, 上のズレ, 下のズレ)
fn rotate_diff1(input: &Globe, crt: &Vec<u16>, r: usize, odd: u32) -> (u8, u8, u8, u8, u8) {
    if input.w == 1 {
        return rotate_diff1_colorful(input, crt, r, odd);
    }
    let Globe {
        rows,
        cols,
        w,
        ref target,
        ..
    } = *input;
    let mut d = 0;
    let mut d2 = 0;
    let mut rot_high = 0;
    let mut rot_low = 0;
    for r in [r, rows - r - 1] {
        let c0 = target[r * cols];
        let c1 = c0 + (cols / w) as u16;
        let mut cnt = [0u8; 66];
        let mut tot = 0;
        for i in 0..cols {
            if c0 <= crt[r * cols + i] && crt[r * cols + i] < c1 {
                tot += 1;
                let p = crt[r * cols + i] - c0;
                let mut s = i + cols - (p + 1) as usize * w + 1;
                if s >= cols {
                    s -= cols;
                }
                cnt[s as usize] += 1;
            }
        }
        let mut sum = 0;
        let mut max = 0;
        let mut max_i = 0;
        if w == 1 {
            for (i, v) in cnt.iter().enumerate() {
                if max.setmax(*v) {
                    max_i = i;
                }
            }
        } else {
            for i in 0..w {
                sum += cnt[cols - 1 - i];
            }
            let mut i2 = cols - w;
            for i in 0..cols {
                sum += cnt[i];
                sum -= cnt[i2];
                if max.setmax(sum) {
                    max_i = i;
                }
                i2 += 1;
                if i2 >= cols {
                    i2 -= cols;
                }
            }
        }
        d += cols - max as usize;
        d2 += cols - max as usize + tot - max as usize;
        if r < rows / 2 {
            rot_high = max_i as u8;
        } else {
            rot_low = max_i as u8;
        }
    }
    let rot = (rot_high as usize + cols - rot_low as usize) % cols;
    (d as u8, d2 as u8, rot as u8, rot_high, rot_low)
}

/// カラフルな場合は偶奇を考慮する必要がある
fn rotate_diff1_colorful(input: &Globe, crt: &Vec<u16>, r: usize, odd: u32) -> (u8, u8, u8, u8, u8) {
    let Globe {
        rows,
        cols,
        w,
        ref target,
        ..
    } = *input;
    let mut d = [[0; 2]; 2];
    let mut d2 = [[0; 2]; 2];
    let mut rot = [[0; 2]; 2];
    for r in [r, rows - r - 1] {
        let c0 = target[r * cols];
        let c1 = c0 + (cols / w) as u16;
        let mut cnt = [0u8; 66];
        let mut tot = 0;
        for i in 0..cols {
            if c0 <= crt[r * cols + i] && crt[r * cols + i] < c1 {
                tot += 1;
                let p = crt[r * cols + i] - c0;
                let mut s = i + cols - (p + 1) as usize * w + 1;
                if s >= cols {
                    s -= cols;
                }
                cnt[s as usize] += 1;
            }
        }
        let mut max = [0; 2];
        let mut max_i = [0, 1];
        for (i, v) in cnt.iter().enumerate() {
            if max[i & 1].setmax(*v) {
                max_i[i & 1] = i;
            }
        }
        let p = if r < rows / 2 { 0 } else { 1 };
        d[p] = [cols - max[0] as usize, cols - max[1] as usize];
        d2[p] = [
            cols - max[0] as usize + tot as usize - max[0] as usize,
            cols - max[1] as usize + tot as usize - max[1] as usize,
        ];
        rot[p] = max_i;
    }
    let odd = (odd >> r & 1) as usize;
    let k = if d[0][0] + d[1][odd] < d[0][1] + d[1][1 - odd] { 0 } else { 1 };
    (
        (d[0][k] + d[1][(k + odd) % 2]) as u8,
        (d2[0][k] + d2[1][(k + odd) % 2]) as u8,
        ((rot[0][k] + cols - rot[1][(k + odd) % 2]) % cols) as u8,
        rot[0][k] as u8,
        rot[1][(k + odd) % 2] as u8,
    )
}

// 色誤差と上下のズレを計算
fn rotate_diff(input: &Globe, crt: &Vec<u16>, odd: u32) -> ([u8; 4], [u8; 4], [u8; 4]) {
    let mut diff = [0; 4];
    let mut diff2 = [0; 4];
    let mut rot = [0; 4];
    for r in 0..input.rows / 2 {
        let a = rotate_diff1(input, crt, r, odd);
        diff[r] = a.0 as u8;
        diff2[r] = a.1 as u8;
        rot[r] = a.2 as u8;
    }
    (diff, diff2, rot)
}

fn vis(input: &Globe, crt: &Vec<u16>) {
    for r in 0..input.rows {
        for c in 0..input.cols {
            eprint!("{:3} ", crt[r * input.cols + c]);
        }
        eprintln!();
    }
    eprintln!();
}

fn shrink(input: &Globe, out: &mut Vec<usize>) {
    loop {
        let mut ok = false;
        let mut i = 0;
        while i + 1 < out.len() {
            if out[i] < input.cols && out[i] == out[i + 1] {
                out.remove(i);
                out.remove(i);
                ok = true;
            } else if out[i] >= input.cols
                && (out[i] == out[i + 1] + input.cols + input.rows || out[i] + input.cols + input.rows == out[i + 1])
            {
                out.remove(i);
                out.remove(i);
                ok = true;
            } else {
                i += 1;
            }
        }
        if !ok {
            break;
        }
    }
}

// 直前の操作の末尾と追加する操作の先頭を打ち消してからつなげる
fn extend_with_overlap(input: &Globe, out: &mut Vec<usize>, add: &[usize]) {
    if add.len() == 0 {
        return;
    } else if out.len() == 0 {
        out.extend(add.iter().cloned());
        return;
    }
    let mut s = out.len() - 1;
    while s >= 1 && out[s] >= input.cols && out[s - 1] >= input.cols {
        s -= 1;
    }
    let mut t = 0;
    while t + 1 < add.len() && add[t] >= input.cols && add[t + 1] >= input.cols {
        t += 1;
    }
    let mut ok = true;
    let mut rem = out[s..].len();
    for i in 0..=t {
        let rev = input.rev(add[i]);
        if let Some(p) = out[s..].iter().position(|&k| k == rev) {
            out.remove(s + p);
            rem -= 1;
        } else {
            ok = false;
            out.push(add[i]);
        }
    }
    if rem > 0 || !ok {
        out.extend(add[t + 1..].iter().cloned());
    } else {
        extend_with_overlap(input, out, &add[t + 1..]);
    }
}

fn overlap(input: &Globe, out: &[usize], add: &[usize]) -> i32 {
    if add.len() == 0 || out.len() == 0 {
        return 0;
    }
    let mut s = out.len() - 1;
    while s >= 1 && out[s] >= input.cols && out[s - 1] >= input.cols {
        s -= 1;
    }
    let mut t = 0;
    while t + 1 < add.len() && add[t] >= input.cols && add[t + 1] >= input.cols {
        t += 1;
    }
    let mut ov = 0;
    let mut tmp = out[s..].to_vec();
    let mut ok = true;
    for i in 0..=t {
        let rev = input.rev(add[i]);
        if let Some(p) = tmp.iter().position(|&k| k == rev) {
            tmp.remove(p);
            ov += 2;
        } else {
            ok = false;
        }
    }
    if tmp.len() > 0 || !ok {
        ov
    } else {
        ov + overlap(input, &out[..s], &add[t + 1..])
    }
}

fn get_loops(dst: &Vec<usize>) -> Vec<Vec<usize>> {
    let mut used = vec![false; dst.len()];
    let mut loops = vec![];
    for i in 0..dst.len() {
        if used[i] {
            continue;
        }
        let mut tmp = vec![];
        let mut v = i;
        while used[v].setmax(true) {
            tmp.push(v);
            v = dst[v];
        }
        if tmp.len() > 1 {
            loops.push(tmp);
        }
    }
    loops
}

fn optimize(input: &Globe, outs: &Vec<Vec<usize>>, rots: &Vec<usize>) -> Vec<usize> {
    profile!(optimize);
    let rows = input.rows;
    let cols = input.cols;
    let m = rows + cols;
    let mut outs = outs.clone();
    for out in &mut outs {
        for op in out {
            if m <= *op && *op < m + cols {
                *op -= m;
            }
        }
    }
    if outs.len() == 2 {
        outs.push(vec![]);
    }
    let mut out = outs[1].clone();
    for r in 2..outs.len() {
        let mut trace = Trace::new();
        let mut dp = mat![(-1, !0); out.len() + 1; outs[r].len() + 1; if r == 2 { cols } else { 1 }; cols];
        dp[0][0][0][0] = (0, !0);
        for i in 0..=out.len() {
            for j in 0..=outs[r].len() {
                for di in 0..dp[i][j].len() {
                    for dj in 0..cols {
                        if dp[i][j][di][dj].0 < 0 {
                            continue;
                        }
                        let (crt, id) = dp[i][j][di][dj];
                        let prev = if dp[i][j][di][dj].1 == !0 {
                            !0
                        } else {
                            trace.prev_move(dp[i][j][di][dj].1)
                        };
                        if i < out.len() && out[i] >= cols {
                            if prev != !0 && (prev + m) % (2 * m) == out[i] {
                                if dp[i + 1][j][di][dj].0.setmax(crt + 2) {
                                    dp[i + 1][j][di][dj].1 = trace.prev(dp[i][j][di][dj].1);
                                }
                            } else if dp[i + 1][j][di][dj].0.setmax(crt) {
                                dp[i + 1][j][di][dj].1 = trace.add(out[i], id);
                            }
                            if r == 2 {
                                let di2 = if out[i] < m { (di + 1) % cols } else { (di + cols - 1) % cols };
                                let mv = m + 2 * cols + rows - 1 - out[i];
                                if prev != !0 && (prev + m) % (2 * m) == mv {
                                    if dp[i + 1][j][di2][dj].0.setmax(crt + 2) {
                                        dp[i + 1][j][di2][dj].1 = trace.prev(dp[i][j][di][dj].1);
                                    }
                                } else if dp[i + 1][j][di2][dj].0.setmax(crt) {
                                    dp[i + 1][j][di2][dj].1 = trace.add(mv, id);
                                }
                            }
                        }
                        if j < outs[r].len() && outs[r][j] >= cols {
                            if prev != !0 && (prev + m) % (2 * m) == outs[r][j] {
                                if dp[i][j + 1][di][dj].0.setmax(crt + 2) {
                                    dp[i][j + 1][di][dj].1 = trace.prev(dp[i][j][di][dj].1);
                                }
                            } else if dp[i][j + 1][di][dj].0.setmax(crt) {
                                dp[i][j + 1][di][dj].1 = trace.add(outs[r][j], id);
                            }
                            let dj2 = if outs[r][j] < m {
                                (dj + 1) % cols
                            } else {
                                (dj + cols - 1) % cols
                            };
                            let mv = m + 2 * cols + rows - 1 - outs[r][j];
                            if prev != !0 && (prev + m) % (2 * m) == mv {
                                if dp[i][j + 1][di][dj2].0.setmax(crt + 2) {
                                    dp[i][j + 1][di][dj2].1 = trace.prev(dp[i][j][di][dj].1);
                                }
                            } else if dp[i][j + 1][di][dj2].0.setmax(crt) {
                                dp[i][j + 1][di][dj2].1 = trace.add(mv, id);
                            }
                        }
                        if i < out.len()
                            && j < outs[r].len()
                            && out[i] < cols
                            && outs[r][j] < cols
                            && (out[i] + di) % cols == (outs[r][j] + dj) % cols
                        {
                            let mut i2 = i + 1;
                            while out[i2] >= cols {
                                i2 += 1;
                            }
                            assert_eq!(out[i], out[i2]);
                            let mut j2 = j + 1;
                            while outs[r][j2] >= cols {
                                j2 += 1;
                            }
                            assert_eq!(outs[r][j], outs[r][j2]);
                            if dp[i2 + 1][j2 + 1][di][dj].0.setmax(crt + 2) {
                                let mut id = trace.add((out[i] + di) % cols, id);
                                for p in i + 1..i2 {
                                    id = trace.add(out[p], id);
                                }
                                for p in j + 1..j2 {
                                    id = trace.add(outs[r][p], id);
                                }
                                id = trace.add((out[i] + di) % cols, id);
                                dp[i2 + 1][j2 + 1][di][dj].1 = id;
                            }
                        }
                        if i < out.len() && out[i] < cols {
                            let mut i2 = i + 1;
                            while out[i2] >= cols {
                                i2 += 1;
                            }
                            assert_eq!(out[i], out[i2]);
                            if dp[i2 + 1][j][di][dj].0.setmax(crt) {
                                let mut id = trace.add((out[i] + di) % cols, id);
                                for p in i + 1..i2 {
                                    id = trace.add(out[p], id);
                                }
                                id = trace.add((out[i] + di) % cols, id);
                                dp[i2 + 1][j][di][dj].1 = id;
                            }
                        }
                        if j < outs[r].len() && outs[r][j] < cols {
                            let mut j2 = j + 1;
                            while outs[r][j2] >= cols {
                                j2 += 1;
                            }
                            assert_eq!(outs[r][j], outs[r][j2]);
                            if dp[i][j2 + 1][di][dj].0.setmax(crt) {
                                let mut id = trace.add((outs[r][j] + dj) % cols, id);
                                for p in j + 1..j2 {
                                    id = trace.add(outs[r][p], id);
                                }
                                id = trace.add((outs[r][j] + dj) % cols, id);
                                dp[i][j2 + 1][di][dj].1 = id;
                            }
                        }
                    }
                }
            }
        }
        let mut min = 100000000;
        if r == 2 {
            let mut min_dij = (0, 0);
            for di in 0..cols {
                for dj in 0..cols {
                    if dp[out.len()][outs[r].len()][di][dj].0 < 0 {
                        continue;
                    }
                    let mut cost = -dp[out.len()][outs[r].len()][di][dj].0;
                    if rots.len() < 4 {
                        for (r, rot) in [(0, di), (rows - 1, di)] {
                            let rot = (rots[r] + rot) % cols;
                            cost += rot.min(cols - rot) as i32;
                        }
                    } else {
                        for (r, rot) in [(0, di), (rows - 1, di), (1, dj), (rows - 2, dj)] {
                            let rot = (rots[r] + rot) % cols;
                            cost += rot.min(cols - rot) as i32;
                        }
                    }
                    if min.setmin(cost) {
                        min_dij = (di, dj);
                    }
                }
            }
            let (di, dj) = min_dij;
            let exp = out.len() + outs[r].len() + min as usize;
            out = trace.get(dp[out.len()][outs[r].len()][di][dj].1);
            if rots.len() < 4 {
                for (r, rot) in [(0, di), (rows - 1, di)] {
                    let rot = (rots[r] + rot) % cols;
                    if rot < cols - rot {
                        for _ in 0..rot {
                            out.push(cols + r);
                        }
                    } else {
                        for _ in 0..cols - rot {
                            out.push(m + cols + r);
                        }
                    }
                }
            } else {
                for (r, rot) in [(0, di), (rows - 1, di), (1, dj), (rows - 2, dj)] {
                    let rot = (rots[r] + rot) % cols;
                    if rot < cols - rot {
                        for _ in 0..rot {
                            out.push(cols + r);
                        }
                    } else {
                        for _ in 0..cols - rot {
                            out.push(m + cols + r);
                        }
                    }
                }
            }
            assert_eq!(out.len(), exp);
        } else {
            let mut min_dj = 0;
            for dj in 0..cols {
                if dp[out.len()][outs[r].len()][0][dj].0 < 0 {
                    continue;
                }
                let mut cost = -dp[out.len()][outs[r].len()][0][dj].0;
                for (r, rot) in [(r - 1, dj), (rows - r, dj)] {
                    let rot = (rots[r] + rot) % cols;
                    cost += rot.min(cols - rot) as i32;
                }
                if min.setmin(cost) {
                    min_dj = dj;
                }
            }
            let dj = min_dj;
            let exp = out.len() + outs[r].len() + min as usize;
            out = trace.get(dp[out.len()][outs[r].len()][0][dj].1);
            for (r, rot) in [(r - 1, dj), (rows - r, dj)] {
                let rot = (rots[r] + rot) % cols;
                if rot < cols - rot {
                    for _ in 0..rot {
                        out.push(cols + r);
                    }
                } else {
                    for _ in 0..cols - rot {
                        out.push(m + cols + r);
                    }
                }
            }
            assert_eq!(out.len(), exp);
        }
    }
    let out = outs[0].iter().cloned().chain(out).collect_vec();
    out
}
