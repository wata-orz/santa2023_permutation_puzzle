#![allow(non_snake_case)]

// 最終パートで1手先読み貪欲でなくビームサーチをする

use duct::cmd;
use itertools::Itertools;
use rand::prelude::*;
use rayon::prelude::*;
use solution::*;
use std::collections::{HashMap, HashSet, VecDeque};
use tools::inv_perm;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColorType {
    Standard,
    Colorful,
    Stripe,
}

#[derive(Clone, Debug)]
struct Input {
    n: usize,
    n2: usize,
    m: usize,
    moves: Vec<Vec<usize>>,
    start: Vec<u16>,
    target: Vec<u16>,
    color_type: ColorType,
    clusters: Vec<Cluster>,
    rot3: Rot3,
    rotn: Vec<Vec<Vec<usize>>>,
}

impl Input {
    fn new(input: &tools::Input) -> Option<Self> {
        if !input.puzzle_type.starts_with("cube") {
            return None;
        }
        let color_type = if input.target == (0..6).map(|i| (0..input.n / 6).map(move |_| i)).flatten().collect_vec() {
            ColorType::Standard
        } else if input.target == (0..input.n as u16).collect_vec() {
            ColorType::Colorful
        } else {
            ColorType::Stripe
        };
        let n = f64::sqrt((input.n / 6) as f64).round() as usize;
        if n <= 3 {
            return None;
        }
        let n2 = n * n;
        let clusters = gen_clusters(n);
        eprintln!("n = {}", n);
        eprintln!("#clusters = {}", clusters.len());
        let rot3 = Rot3::new(n);
        // 高速化のために、各操作を巡回置換に分解しておく
        let mut rotn = vec![vec![]; input.m * 2];
        for k in 0..input.m * 2 {
            rotn[k] = get_loops(&input.moves[k]).into_iter().filter(|a| a.len() > 1).collect_vec();
        }
        Some(Input {
            n,
            n2,
            m: input.m,
            moves: input.moves.clone(),
            start: input.start.clone(),
            target: input.target.clone(),
            color_type,
            clusters,
            rot3,
            rotn,
        })
    }
}

fn main() {
    get_time();
    let input0 = read_input();
    let Some(mut input) = Input::new(&input0) else {
        return;
    };
    let is_local = false;
    let n = input.n;
    if is_local && n == 33 {
        return;
    }
    let mut start0 = input.start.clone();
    let mut out = vec![];
    // まず最初に角、中央、中央エッジを揃える
    if n % 2 != 0 {
        if input.color_type == ColorType::Stripe {
            input.target = (0..6).map(|i| (0..input.n2).map(move |_| i)).flatten().collect_vec();
            input.start = (0..6)
                .map(|i| {
                    (0..input.n2)
                        .map(|j| ((input.start[i * input.n2 + j] + 6 - j as u16 % 2) % 6))
                        .collect_vec()
                })
                .flatten()
                .collect_vec();
            input.color_type = ColorType::Standard;
            start0 = input.start.clone();
        }
        let f = format!("{}/cache/corner/{:04}.txt", env!("CARGO_MANIFEST_DIR"), input0.id);
        if let Ok(s) = std::fs::read_to_string(&f) {
            eprintln!("read from file");
            out = tools::parse_output(&input0, &s).unwrap().out;
            for &mv in &out {
                input.start = apply(&input.start, &input0.moves[mv]);
            }
        } else {
            let mut crt1 = input.start.iter().skip(n * n / 2).step_by(n * n).cloned().collect_vec();
            if input.color_type == ColorType::Colorful {
                crt1 = crt1.into_iter().map(|x| x / (n * n) as u16).collect_vec();
            }
            let out1 = solve1(&crt1);
            for mv in out1 {
                out.push(mv * n + n / 2);
                input.start = apply(&input.start, &input0.moves[mv * n + n / 2]);
            }
            let mut crt3 = vec![];
            for k in 0..6 {
                for i in 0..3 {
                    for j in 0..3 {
                        let mut v = input.start[k * n * n + (i * (n / 2)) * n + j * (n / 2)];
                        if input.color_type == ColorType::Colorful {
                            v /= (n * n) as u16;
                        }
                        crt3.push(v);
                    }
                }
            }
            let out3 = solve_small(&crt3, 3);
            for mv in out3 {
                let mv = (mv / 3) * n + (mv % 3) * (n / 2);
                out.push(mv);
                input.start = apply(&input.start, &input0.moves[mv]);
            }
        }
    } else {
        let mut crt2 = vec![];
        let mut target2 = vec![];
        for k in 0..6 {
            for i in 0..2 {
                for j in 0..2 {
                    let mut v = input.start[k * n * n + (i * (n - 1)) * n + j * (n - 1)];
                    if input.color_type == ColorType::Colorful {
                        v /= (n * n) as u16;
                    }
                    crt2.push(v);
                    let mut v = input.target[k * n * n + (i * (n - 1)) * n + j * (n - 1)];
                    if input.color_type == ColorType::Colorful {
                        v /= (n * n) as u16;
                    }
                    target2.push(v);
                }
            }
        }
        let out2: Vec<usize> = solve_bfs(&crt2, &target2, &MOVES2.iter().map(|mv| mv.to_vec()).collect_vec());
        for mv in out2 {
            let mv = (mv / 2) * n + (mv % 2) * (n - 1);
            out.push(mv);
            input.start = apply(&input.start, &input0.moves[mv]);
        }
    }
    if n <= 3 {
        write_output(&input0, &out);
        return;
    }
    let f = format!(
        "{}/cache/init/{:04}_{}_{}_v5.txt",
        env!("CARGO_MANIFEST_DIR"),
        input0.id,
        AVG_INIT,
        OPPOSITE_WEIGHT
    );
    if let Ok(s) = std::fs::read_to_string(&f) {
        out = tools::parse_output(&input0, &s).unwrap().out;
        let mut start = start0.clone();
        apply_loops(&input, &mut start, &out);
        input.start = start;
    } else {
        init_optimize(&mut input, &mut out);
        std::fs::write(&f, out.iter().map(|&mv| &input0.move_names[mv]).join(".")).unwrap();
    }
    if let Ok(s) = std::env::var("RESTART") {
        let f = format!("{}/{}/{:04}.txt", env!("CARGO_MANIFEST_DIR"), s, input0.id);
        out = tools::parse_output(&input0, &std::fs::read_to_string(&f).unwrap())
            .unwrap()
            .out;
        let mut start = start0.clone();
        apply_loops(&input, &mut start, &out);
        input.start = start;
    }
    // cids[x] = (c, i) は x がクラスター c の i 番目の頂点であることを表す
    let mut cids0 = vec![(!0, !0); 6 * input.n2];
    for c in 0..input.clusters.len() {
        match input.clusters[c] {
            Cluster::Edge(ref vs) => {
                for (i, &(v, _)) in vs.iter().enumerate() {
                    cids0[v] = (c, i);
                }
            }
            Cluster::Inner(ref vs, _) => {
                for (i, &v) in vs.iter().enumerate() {
                    cids0[v] = (c, i);
                }
            }
        }
    }

    let avg = 1.75;
    let n2 = input.n2 as u16;
    let mut last_time = get_time();
    let mut last_large_move = 0;
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(5340980);
    let mut moves = vec![];
    for i in 1..n - 1 {
        if i * 2 + 1 == n {
            continue;
        }
        moves.push(i);
        moves.push(i + n);
        moves.push(i + n * 2);
    }
    let mut rot3_3 = mat![vec![]; input.m * 3; input.m * 3];
    {
        let mut crt = (0..input0.n as u16).collect_vec();
        let mut mv = vec![];
        let mut used = vec![false; input0.n];
        for k1 in 0..input.m * 3 {
            if k1 % n == 0 || k1 % n == n - 1 {
                continue;
            }
            for k2 in 0..input.m * 3 {
                if k2 % n == 0
                    || k2 % n == n - 1
                    || (k1 / n) % 3 == (k2 / n) % 3
                    || (k1 % n) * 2 + 1 == n && (k2 % n) * 2 + 1 == n
                {
                    continue;
                }
                mv.clear();
                if k1 >= input.m * 2 {
                    mv.push(k1 - input.m * 2);
                    mv.push(k1 - input.m * 2);
                } else {
                    mv.push(k1);
                }
                if k2 >= input.m * 2 {
                    mv.push(k2 - input.m * 2);
                    mv.push(k2 - input.m * 2);
                } else {
                    mv.push(k2);
                }
                if k1 >= input.m * 2 {
                    mv.push(k1 - input.m);
                    mv.push(k1 - input.m);
                } else {
                    mv.push((k1 + input.m) % (input.m * 2));
                }
                if k2 >= input.m * 2 {
                    mv.push(k2 - input.m);
                    mv.push(k2 - input.m);
                } else {
                    mv.push((k2 + input.m) % (input.m * 2));
                }
                for i in 0..input0.n {
                    crt[i] = i as u16;
                }
                apply_loops(&input, &mut crt, &mv);
                used.fill(false);
                for i in 0..input0.n {
                    if used[i] || crt[i] == i as u16 {
                        continue;
                    }
                    let mut tmp = vec![];
                    let mut v = i;
                    while used[v].setmax(true) {
                        tmp.push(v);
                        v = crt[v] as usize;
                    }
                    rot3_3[k1][k2].push(tmp);
                }
                assert!(
                    rot3_3[k1][k2].len() == 2
                        && (rot3_3[k1][k2][0].len() == 3 && rot3_3[k1][k2][1].len() == 3
                            || rot3_3[k1][k2][0].len() == 2 && rot3_3[k1][k2][1].len() == 2)
                );
            }
        }
    }
    for iter in 0.. {
        if get_time() - last_time > 1000.0 {
            last_time = get_time();
            write_output(&input0, &out);
        }
        let mut start = start0.clone();
        apply_loops(&input, &mut start, &out);
        input.start = start;
        let crt_diff = diff2(&input, &input.start);
        eprintln!(
            "{:.3}: len = {}, diff = {}",
            get_time(),
            out.len(),
            diff2(&input, &input.start),
        );
        if diff(&input.start, &input.target) <= input0.wildcard {
            break;
        }
        let mut min = (2.0, 0.0, 10, 0.0);
        let mut best_out = vec![];
        const C: usize = 100;
        let mut target = input.target.clone();
        for &op in out.iter().rev() {
            apply_loop(&input, &mut target, (op + input.m) % (input.m * 2));
        }
        let mut start = start0.clone();
        let mut starts = vec![];
        let mut targets = vec![];
        let mut seeds = vec![];
        for t in 0..=out.len() {
            starts.push(start.clone());
            targets.push(target.clone());
            seeds.push(rng.gen());
            if t == out.len() {
                break;
            }
            apply_loop(&input, &mut target, out[t]);
            apply_loop(&input, &mut start, out[t]);
        }
        let ret = (0..=out.len())
            .into_par_iter()
            .map(|t| {
                let mut rng = rand_pcg::Pcg64Mcg::from_seed(seeds[t]);
                let mut best = vec![((0.0, -1000), !0, vec![]); C];
                let mut start = starts[t].clone();
                let mut input = input.clone();
                input.target = targets[t].clone();
                // TODO: 逆向きにも対応したほうが良い？
                // 直前の操作を打ち消す場合に対応するために、直前にどの方向にどの層を動かしたかを記憶しておく
                let mut prev_is = 0usize;
                let prev_dir = if t == 0 { !0 } else { out[t - 1] / n };
                for t in (0..t).rev() {
                    if out[t] / n != prev_dir {
                        break;
                    }
                    prev_is |= 1 << (out[t] % n);
                }
                // 直後の操作を打ち消す場合に対応するために、直後にどの方向にどの層を動かしたかを記憶しておく
                let mut next_is = 0usize;
                let next_dir = if t == out.len() { !0 } else { out[t] / n };
                for t in t..out.len() {
                    if out[t] / n != next_dir {
                        break;
                    }
                    next_is |= 1 << (out[t] % n);
                }
                if iter - last_large_move < 100 {
                    // [di, ri] や [di, r(n-1-i)] は偶奇を壊さないので使用可能
                    for d1 in 0..6 {
                        for d2 in 0..6 {
                            if d1 % 3 == d2 % 3 {
                                continue;
                            }
                            for i in 1..n / 2 {
                                for i1 in [i, n - 1 - i] {
                                    for i2 in [i, n - 1 - i] {
                                        let k1 = d1 * n + i1;
                                        let k2 = d2 * n + i2;
                                        let ov = if (prev_dir + 3) % 6 == d1 && prev_is >> i1 & 1 != 0 {
                                            2
                                        } else {
                                            0
                                        } + if (next_dir + 3) % 6 == d2 && next_is >> i2 & 1 != 0 {
                                            2
                                        } else {
                                            0
                                        };
                                        let diff = (apply_loops(&input, &mut start, &[k1, k2]) * avg + (2 - ov) as f64, -3);
                                        if best[C - 1].0 > diff {
                                            best[C - 1] = (diff, t, vec![k1, k2]);
                                            for c in (0..C - 1).rev() {
                                                if best[c].0 > best[c + 1].0 {
                                                    best.swap(c, c + 1);
                                                } else {
                                                    break;
                                                }
                                            }
                                        }
                                        apply_loops(
                                            &input,
                                            &mut start,
                                            &[(k2 + input.m) % (2 * input.m), (k1 + input.m) % (2 * input.m)],
                                        );
                                    }
                                }
                            }
                        }
                    }
                    // [di, di]
                    for &k in &moves {
                        let ov = if (prev_dir + 3) % 6 == k / n && prev_is >> (k % n) & 1 != 0 {
                            2
                        } else {
                            0
                        } + if (next_dir + 3) % 6 == k / n && next_is >> (k % n) & 1 != 0 {
                            2
                        } else {
                            0
                        };
                        let diff = (apply_loops(&input, &mut start, &[k, k]) * avg + (2 - ov) as f64, -3);
                        if best[C - 1].0 > diff {
                            best[C - 1] = (diff, t, vec![k, k]);
                            for c in (0..C - 1).rev() {
                                if best[c].0 > best[c + 1].0 {
                                    best.swap(c, c + 1);
                                } else {
                                    break;
                                }
                            }
                        }
                        apply_loops(
                            &input,
                            &mut start,
                            &[(k + input.m) % (2 * input.m), (k + input.m) % (2 * input.m)],
                        );
                    }
                    // [d0, ri, -d0, -ri]
                    for i in 0..input.m * 2 {
                        for j in 0..input.m * 2 {
                            if (i % n != 0 && i % n != n - 1) == (j % n != 0 && j % n != n - 1) || i / n % 3 == j / n % 3 {
                                continue;
                            }
                            let mv = vec![i, j, (i + input.m) % (input.m * 2), (j + input.m) % (input.m * 2)];
                            let ov = if (prev_dir + 3) % 6 == i / n && prev_is >> (i % n) & 1 != 0 {
                                2
                            } else {
                                0
                            } + if next_dir == j / n && next_is >> (j % n) & 1 != 0 {
                                2
                            } else {
                                0
                            };
                            let diff = (apply_loops(&input, &mut start, &mv) * avg + (4 - ov) as f64, -3);
                            if best[C - 1].0 > diff {
                                best[C - 1] = (diff, t, mv);
                                for c in (0..C - 1).rev() {
                                    if best[c].0 > best[c + 1].0 {
                                        best.swap(c, c + 1);
                                    } else {
                                        break;
                                    }
                                }
                            }
                            apply_loops(
                                &input,
                                &mut start,
                                &[j, i, (j + input.m) % (input.m * 2), (i + input.m) % (input.m * 2)],
                            );
                        }
                    }
                }
                // 3-rot×2を矩形に対して一気に適用するパターン [d1, d2, r1, r2, -d1, -d2, -r1, -r2]
                for d1 in 0..9 {
                    for d2 in 0..9 {
                        if d1 % 3 == d2 % 3 {
                            continue;
                        }
                        let mut delta = mat![0.0; n; n];
                        for i in 1..n - 1 {
                            for j in 1..n - 1 {
                                if i * 2 + 1 == n && j * 2 + 1 == n {
                                    continue;
                                }
                                let k1 = d1 * n + i;
                                let k2 = d2 * n + j;
                                for r in &rot3_3[k1][k2] {
                                    if r.len() == 3 {
                                        let v0 = start[r[0]];
                                        let v1 = start[r[1]];
                                        let v2 = start[r[2]];
                                        let u0 = input.target[r[0]];
                                        let u1 = input.target[r[1]];
                                        let u2 = input.target[r[2]];
                                        if v0 != u0 {
                                            if input.color_type == ColorType::Colorful {
                                                if v0 / n2 + u0 / n2 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            } else {
                                                if v0 + u0 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            }
                                        }
                                        if v1 != u1 {
                                            if input.color_type == ColorType::Colorful {
                                                if v1 / n2 + u1 / n2 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            } else {
                                                if v1 + u1 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            }
                                        }
                                        if v2 != u2 {
                                            if input.color_type == ColorType::Colorful {
                                                if v2 / n2 + u2 / n2 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            } else {
                                                if v2 + u2 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            }
                                        }
                                        if v0 != u2 {
                                            if input.color_type == ColorType::Colorful {
                                                if v0 / n2 + u2 / n2 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            } else {
                                                if v0 + u2 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            }
                                        }
                                        if v1 != u0 {
                                            if input.color_type == ColorType::Colorful {
                                                if v1 / n2 + u0 / n2 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            } else {
                                                if v1 + u0 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            }
                                        }
                                        if v2 != u1 {
                                            if input.color_type == ColorType::Colorful {
                                                if v2 / n2 + u1 / n2 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            } else {
                                                if v2 + u1 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            }
                                        }
                                    } else {
                                        let v0 = start[r[0]];
                                        let v1 = start[r[1]];
                                        let u0 = input.target[r[0]];
                                        let u1 = input.target[r[1]];
                                        if v0 != u0 {
                                            if input.color_type == ColorType::Colorful {
                                                if v0 / n2 + u0 / n2 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            } else {
                                                if v0 + u0 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            }
                                        }
                                        if v1 != u1 {
                                            if input.color_type == ColorType::Colorful {
                                                if v1 / n2 + u1 / n2 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            } else {
                                                if v1 + u1 == 5 {
                                                    delta[i][j] -= avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] -= avg;
                                                }
                                            }
                                        }
                                        if v0 != u1 {
                                            if input.color_type == ColorType::Colorful {
                                                if v0 / n2 + u1 / n2 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            } else {
                                                if v0 + u1 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            }
                                        }
                                        if v1 != u0 {
                                            if input.color_type == ColorType::Colorful {
                                                if v1 / n2 + u0 / n2 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            } else {
                                                if v1 + u0 == 5 {
                                                    delta[i][j] += avg * OPPOSITE_WEIGHT;
                                                } else {
                                                    delta[i][j] += avg;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        let mut js = 0;
                        let mut crt = 0.0;
                        const T0: f64 = 10.0;
                        const T1: f64 = 0.0;
                        const REP: usize = 2000;
                        let mut min = 0.0;
                        let mut min_is = 0usize;
                        let mut min_js = 0usize;
                        for iter in 0..REP {
                            let p = rng.gen_range(1..n - 1);
                            let bk = js;
                            js ^= 1 << p;
                            if d1 >= 6 && p * 2 + 1 != 0 && js >> p & 1 != 0 && js >> (n - 1 - p) & 1 != 0 {
                                js ^= 1 << (n - 1 - p);
                            }
                            let mut sum = if d2 < 6 { 2.0 } else { 4.0 } * usize::count_ones(js) as f64;
                            if next_dir == d2 {
                                for j in 1..n - 1 {
                                    if js >> j & 1 != 0 && next_is >> j & 1 != 0 {
                                        sum -= 2.0;
                                    }
                                }
                            }
                            let mut is = 0usize;
                            if d2 < 6 {
                                for i in 1..n - 1 {
                                    if i * 2 + 1 == n && js >> i & 1 != 0 {
                                        continue;
                                    }
                                    let mut tmp = if d1 < 6 {
                                        if (prev_dir + 3) % 6 == d1 && prev_is >> i & 1 != 0 {
                                            0.0
                                        } else {
                                            2.0
                                        }
                                    } else {
                                        4.0
                                    };
                                    for j in 1..n - 1 {
                                        if js >> j & 1 != 0 {
                                            tmp += delta[i][j];
                                        }
                                    }
                                    if tmp < 0.0 {
                                        sum += tmp;
                                        is |= 1 << i;
                                    }
                                }
                            } else {
                                for i in 1..n / 2 {
                                    if i * 2 + 1 == n && js >> i & 1 != 0 {
                                        continue;
                                    }
                                    let mut tmp1 = if d1 < 6 {
                                        if (prev_dir + 3) % 6 == d1 && prev_is >> i & 1 != 0 {
                                            0.0
                                        } else {
                                            2.0
                                        }
                                    } else {
                                        4.0
                                    };
                                    for j in 1..n - 1 {
                                        if js >> j & 1 != 0 {
                                            tmp1 += delta[i][j];
                                        }
                                    }
                                    let mut tmp2 = if d1 < 6 {
                                        if (prev_dir + 3) % 6 == d1 && prev_is >> (n - 1 - i) & 1 != 0 {
                                            0.0
                                        } else {
                                            2.0
                                        }
                                    } else {
                                        4.0
                                    };
                                    for j in 1..n - 1 {
                                        if js >> j & 1 != 0 {
                                            tmp2 += delta[n - 1 - i][j];
                                        }
                                    }
                                    if tmp1 < tmp2 && tmp1 < 0.0 {
                                        sum += tmp1;
                                        is |= 1 << i;
                                    } else if tmp2 < 0.0 {
                                        sum += tmp2;
                                        is |= 1 << (n - 1 - i);
                                    }
                                }
                            }
                            let time = iter as f64 / REP as f64;
                            let temp = T0 * (1.0 - time) + T1 * time;
                            if crt >= sum || rng.gen_bool(f64::exp((crt - sum) / temp)) {
                                crt = sum;
                                if min.setmin(crt) {
                                    min_is = is;
                                    min_js = js;
                                }
                            } else {
                                js = bk;
                            }
                        }
                        let min = (
                            min,
                            if min_is.count_ones() == 1 && min_js.count_ones() == 1 {
                                -1
                            } else {
                                -2
                            },
                        );
                        if best[C - 1].0 > min {
                            let is = min_is;
                            let js = min_js;
                            let mut out2 = vec![];
                            for i in 1..n - 1 {
                                if is >> i & 1 != 0 {
                                    if d1 < 6 {
                                        out2.push(d1 * n + i);
                                    } else {
                                        out2.push((d1 % 6) * n + i);
                                        out2.push((d1 % 6) * n + i);
                                    }
                                }
                            }
                            for j in 1..n - 1 {
                                if js >> j & 1 != 0 {
                                    if d2 < 6 {
                                        out2.push(d2 * n + j);
                                    } else {
                                        out2.push((d2 % 6) * n + j);
                                        out2.push((d2 % 6) * n + j);
                                    }
                                }
                            }
                            for i in 1..n - 1 {
                                if is >> i & 1 != 0 {
                                    if d1 < 6 {
                                        out2.push((d1 * n + i + input.m) % (input.m * 2));
                                    } else {
                                        out2.push(((d1 % 6) * n + i + input.m) % (input.m * 2));
                                        out2.push(((d1 % 6) * n + i + input.m) % (input.m * 2));
                                    }
                                }
                            }
                            for j in 1..n - 1 {
                                if js >> j & 1 != 0 {
                                    if d2 < 6 {
                                        out2.push((d2 * n + j + input.m) % (input.m * 2));
                                    } else {
                                        out2.push(((d2 % 6) * n + j + input.m) % (input.m * 2));
                                        out2.push(((d2 % 6) * n + j + input.m) % (input.m * 2));
                                    }
                                }
                            }
                            best[C - 1] = (min, t, out2);
                            for c in (0..C - 1).rev() {
                                if best[c].0 > best[c + 1].0 {
                                    best.swap(c, c + 1);
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
                best
            })
            .collect::<Vec<_>>();
        let mut best = ret.into_iter().flatten().filter(|a| a.1 != !0).collect_vec();
        best.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        best.truncate(C);
        let best = best
            .into_par_iter()
            .map(|((_, move_type), t, mv)| {
                let mut out2 = out[..t].to_vec();
                extend_with_overlap(&mut out2, &mv, n);
                extend_with_overlap(&mut out2, &out[t..], n);
                let mut start = start0.clone();
                apply_loops(&input, &mut start, &out2);
                let mut input = input.clone();
                input.start = start;
                let d = crt_diff - diff2(&input, &input.start);
                (d, d, move_type, out2)
            })
            .collect::<Vec<_>>();
        for (d, diff_cost, move_type, out2) in best {
            if diff_cost > 0.0
                && min.setmin((
                    (out2.len() as f64 - out.len() as f64) / diff_cost,
                    (out2.len() as f64 - out.len() as f64) / d.max(0.01),
                    move_type,
                    -diff_cost,
                ))
            {
                best_out = out2;
            }
        }
        if min.2 == 10 {
            break;
        }
        if min.2 == -3 {
            last_large_move = iter;
        }
        eprintln!("min = {:?}", min);
        input.start = start;
        out = best_out;
    }

    const C: usize = 1000;
    let mut beam = vec![];
    {
        let (can_use_list, cost) = decompose_clusters(&input);
        let mut cids = cids0.clone();
        for &op in out.iter().rev() {
            apply_fast(&input, &mut cids, (op + input.m) % (input.m * 2));
        }
        let mut cand = BoundedSortedList::new(C);
        let diff = diff2(&input, &input.start);
        let mut tmp = vec![];
        for t in 0..=out.len() {
            let mut ids = mat![!0; input.clusters.len(); 24];
            for x in 0..6 * input.n2 {
                if cids[x].0 != !0 {
                    let (c, i) = cids[x];
                    let (_, j) = cids0[x];
                    ids[c][i] = j;
                }
            }
            for c in 0..can_use_list.len() {
                let cluster = &input.clusters[c];
                for &(i, j, k, w) in &can_use_list[c] {
                    let i2 = ids[c][i];
                    let j2 = ids[c][j];
                    let k2 = ids[c][k];
                    for mv in input.rot3.get(cluster, i2, k2, j2) {
                        let ov = overlap(&out[..t], &mv, n) + overlap(&mv, &out[t..], n);
                        let key = (mv.len() as i32 - ov, w);
                        tmp.push(key);
                        if cand.can_insert(key) {
                            cand.insert(key, Cand { t, c, mv });
                        }
                    }
                }
            }
            if t < out.len() {
                apply_fast(&input, &mut cids, out[t]);
            }
        }
        tmp.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        tmp.truncate(C);
        beam.push(State {
            out: out.clone(),
            cost,
            diff,
            cands: cand.list(),
            can_use_list,
        });
    }
    const B: usize = 100000;
    loop {
        eprintln!(
            "{:.3}: cost = {}, ({}, {}) - ({}, {})",
            get_time(),
            beam[0].cost,
            beam[0].out.len(),
            beam[0].diff,
            beam[beam.len() - 1].out.len(),
            beam[beam.len() - 1].diff
        );
        if let Some(i) = (0..beam.len()).find(|&i| beam[i].diff <= input0.wildcard as f64) {
            out = beam[i].out.clone();
            break;
        }
        let mut cand = BoundedSortedList::new(B);
        for b in 0..beam.len() {
            for c in 0..beam[b].cands.len() {
                let (dl, dd) = beam[b].cands[c].0;
                cand.insert((beam[b].out.len() as i32 + dl, beam[b].diff + dd), (b, c))
            }
        }
        let mut used = HashSet::new();
        let cand = cand
            .list()
            .into_par_iter()
            .map(|((_, diff), (b, c))| {
                let Cand { t, c, ref mv } = beam[b].cands[c].1;
                let out0 = &beam[b].out;
                let mut out = out0[..t].to_vec();
                extend_with_overlap(&mut out, mv, n);
                extend_with_overlap(&mut out, &beam[b].out[t..], n);
                let mut crt = start0.clone();
                for &mv in &out {
                    apply_fast(&input, &mut crt, mv);
                }
                (crt, out0, out, b, c, diff)
            })
            .collect::<Vec<_>>();
        let mut next = vec![];
        for (crt, out0, out, b, c, diff) in cand {
            if used.insert(crt.clone()) {
                next.push((out0, out, crt, b, c, diff));
            }
        }
        beam = next
            .into_par_iter()
            .map(|(out0, out, crt, b, c, diff)| {
                let t0 = (0..out0.len().min(out.len()))
                    .find(|&t| out0[t] != out[t])
                    .unwrap_or(out0.len().min(out.len()));
                let t1 = out.len()
                    - (0..out0.len().min(out.len()))
                        .find(|&t| out0[out0.len() - 1 - t] != out[out.len() - 1 - t])
                        .unwrap_or(out0.len().min(out.len()));
                let mut cand = BoundedSortedList::new(C);
                for &(key, ref a) in &beam[b].cands {
                    if a.c == c {
                        continue;
                    }
                    if a.t < t0 {
                        cand.insert(key, a.clone());
                    } else if out0.len() + t1 - out.len() < a.t {
                        let mut a = a.clone();
                        a.t += out.len() - out0.len();
                        cand.insert(key, a);
                    }
                }
                let mut can_use_list = beam[b].can_use_list.clone();
                can_use_list[c] = decompose_cluster(&input, &crt, c).0;
                let mut cids = cids0.clone();
                for &op in out.iter().rev() {
                    apply_fast(&input, &mut cids, (op + input.m) % (input.m * 2));
                }
                let cost = beam[b].cost - 1;
                for t in 0..=out.len() {
                    let mut ids = mat![!0; input.clusters.len(); 24];
                    for x in 0..6 * input.n2 {
                        if cids[x].0 != !0 {
                            let (c, i) = cids[x];
                            let (_, j) = cids0[x];
                            ids[c][i] = j;
                        }
                    }
                    {
                        for c in if t < t0 || t1 < t { c..c + 1 } else { 0..input.clusters.len() } {
                            let cluster = &input.clusters[c];
                            for &(i, j, k, w) in &can_use_list[c] {
                                let i2 = ids[c][i];
                                let j2 = ids[c][j];
                                let k2 = ids[c][k];
                                for mv in input.rot3.get(cluster, i2, k2, j2) {
                                    let ov = overlap(&out[..t], &mv, n) + overlap(&mv, &out[t..], n);
                                    let key = (mv.len() as i32 - ov, w);
                                    if cand.can_insert(key) {
                                        cand.insert(key, Cand { t, c, mv });
                                    }
                                }
                            }
                        }
                    }
                    if t < out.len() {
                        apply_fast(&input, &mut cids, out[t]);
                    }
                }
                State {
                    out,
                    cost,
                    diff,
                    cands: cand.list(),
                    can_use_list,
                }
            })
            .collect::<Vec<_>>();
    }
    if is_local || input0.wildcard > 0 {
        write_output(&input0, &optimize_bisearch(&input0, &out));
    } else {
        write_output(&input0, &out);
    }
}

#[derive(Clone, Debug)]
struct State {
    out: Vec<usize>,
    cost: i32,
    diff: f64,
    cands: Vec<((i32, f64), Cand)>,
    can_use_list: Vec<Vec<(usize, usize, usize, f64)>>,
}

#[derive(Clone, Debug)]
struct Cand {
    t: usize,
    c: usize,
    mv: Vec<usize>,
}

const AVG_INIT: f64 = 1.5;

fn init_optimize(input: &mut Input, out: &mut Vec<usize>) {
    let n = input.n;
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(89043280);
    let mut start = input.start.clone();
    for &op in out.iter().rev() {
        apply_loop(input, &mut start, (op + input.m) % (input.m * 2));
    }
    let clusters = if input.color_type == ColorType::Colorful {
        input.clusters.clone()
    } else {
        input
            .clusters
            .iter()
            .filter(|c| matches!(c, Cluster::Edge(_)))
            .cloned()
            .collect_vec()
    };
    let mut moves = vec![];
    for i in 1..n - 1 {
        if i * 2 + 1 == n {
            continue;
        }
        moves.push(i);
        moves.push(i + n);
        moves.push(i + n * 2);
    }
    eprintln!("matrix size: {} x {}", clusters.len(), moves.len());
    define_mod!(P; 2);
    let mut A = Mat(mat![P::new(0); clusters.len(); moves.len()]);
    let mut b = vec![P::new(0); clusters.len()];
    for i in 0..clusters.len() {
        for j in 0..moves.len() {
            let flip = match clusters[i] {
                Cluster::Edge(ref vs) => vs.iter().filter(|&&(v, _)| input.moves[moves[j]][v] != v).count(),
                Cluster::Inner(ref vs, _) => vs.iter().filter(|&&v| input.moves[moves[j]][v] != v).count(),
            };
            if flip == 4 {
                A[i][j] = P::new(1);
            }
        }
        let dst = get_dst(&clusters[i], &input.start, &input.target);
        if is_odd(&dst) {
            b[i] = P::new(1);
        }
    }
    let x = A.gauss().solve(b);
    assert!(x.len() > 0);
    {
        for i in 0..moves.len() {
            if x[0][i].0 != 0 {
                out.push(moves[i]);
            }
        }
    }
    dbg!(out.len());
    const TL: i32 = 20000000;
    let T0 = 3.0;
    let T1 = 0.0;
    let mut score = apply_loops(&input, &mut start.clone(), &out) as f64 * AVG_INIT + out.len() as f64;
    for t in 0..TL {
        if t % (TL / 100) == 0 {
            eprintln!("{:.2}: {}", t as f64 / TL as f64, score);
        }
        let t = t as f64 / TL as f64;
        let temp = T0 * (1.0 - t) + T1 * t;
        match rng.gen_range(0..4) {
            0 => {
                // 1つの操作をランダムに変更
                if out.len() == 0 {
                    continue;
                }
                let i = rng.gen_range(0..out.len());
                let v = out[i] % n;
                if v == 0 || v == n - 1 || v * 2 + 1 == n {
                    continue;
                }
                let bk = out[i];
                out[i] = [v, n - 1 - v, v + n, n * 2 - 1 - v, v + n * 2, n * 3 - 1 - v][rng.gen_range(0..6)]
                    + rng.gen_range(0..=1) * n * 3;
                let score2 = apply_loops(&input, &mut start.clone(), &out) as f64 * AVG_INIT + out.len() as f64;
                if score >= score2 || rng.gen_bool(f64::exp((score - score2) / temp)) {
                    score = score2;
                } else {
                    out[i] = bk;
                }
            }
            1 => {
                // 隣接する2つの操作を入れ替える
                if out.len() <= 1 {
                    continue;
                }
                let i = rng.gen_range(0..out.len() - 1);
                if (out[i] % n == 0 || out[i] % n == n - 1 || out[i] % n * 2 + 1 == n)
                    && (out[i + 1] % n == 0 || out[i + 1] % n == n - 1 || out[i + 1] % n * 2 + 1 == n)
                {
                    continue;
                }
                out.swap(i, i + 1);
                let score2 = apply_loops(&input, &mut start.clone(), &out) as f64 * AVG_INIT + out.len() as f64;
                if score >= score2 || rng.gen_bool(f64::exp((score - score2) / temp)) {
                    score = score2;
                } else {
                    out.swap(i, i + 1);
                }
            }
            2 => {
                // 偶奇を崩さないように2操作を追加
                let v = rng.gen_range(1..n - 1);
                if v * 2 + 1 == n {
                    continue;
                }
                let (i, j) = if rng.gen_bool(0.1) {
                    (out.len(), out.len() + 1)
                } else {
                    (rng.gen_range(0..out.len() + 1), rng.gen_range(0..out.len() + 2))
                };
                let mut crt2 = out.clone();
                let mv = [v, n - 1 - v, v + n, n * 2 - 1 - v, v + n * 2, n * 3 - 1 - v][rng.gen_range(0..6)]
                    + rng.gen_range(0..=1) * n * 3;
                crt2.insert(i, mv);
                let mv2 = if true || rng.gen_bool(0.5) {
                    [v, n - 1 - v, v + n, n * 2 - 1 - v, v + n * 2, n * 3 - 1 - v][rng.gen_range(0..6)]
                        + rng.gen_range(0..=1) * n * 3
                } else {
                    (mv + 3 * n) % (6 * n)
                };
                crt2.insert(j, mv2);
                let score2 = apply_loops(&input, &mut start.clone(), &crt2) as f64 * AVG_INIT + crt2.len() as f64;
                if score >= score2 || rng.gen_bool(f64::exp((score - score2) / temp)) {
                    score = score2;
                    *out = crt2;
                }
            }
            3 => {
                // 偶奇数を崩さないように2操作を削除
                if out.len() < 2 {
                    continue;
                }
                let i = rng.gen_range(0..out.len());
                if out[i] % n == 0 || out[i] % n == n - 1 || out[i] % n * 2 + 1 == n {
                    continue;
                }
                let Some(j) = (0..out.len()).filter(|&j| i != j && (out[i] % n == out[j] % n || out[i] % n + out[j] % n == n - 1)).choose(&mut rng) else {
                    continue;
                };
                let mut crt2 = out.clone();
                crt2.remove(i.max(j));
                crt2.remove(i.min(j));
                let score2 = apply_loops(&input, &mut start.clone(), &crt2) as f64 * AVG_INIT + crt2.len() as f64;
                if score >= score2 || rng.gen_bool(f64::exp((score - score2) / temp)) {
                    score = score2;
                    *out = crt2;
                }
            }
            _ => unreachable!(),
        }
    }
    eprintln!("{:.2}: {}", 1.0, score);
    apply_loops(&input, &mut start, &out);
    input.start = start;
}

fn decompose_cluster(input: &Input, start: &[u16], c: usize) -> (Vec<(usize, usize, usize, f64)>, i32) {
    let mut sum = 0;
    let mut list = vec![];
    let mut g = vec![vec![]; 24];
    let cluster = &input.clusters[c];
    match &cluster {
        Cluster::Edge(vs) => {
            for i in 0..24 {
                for j in 0..24 {
                    if start[vs[i].0] == input.target[vs[j].0] && start[vs[i].1] == input.target[vs[j].1] {
                        g[i].push(j);
                    }
                }
            }
        }
        Cluster::Inner(vs, _) => {
            for i in 0..24 {
                for j in 0..24 {
                    if start[vs[i]] == input.target[vs[j]] {
                        g[i].push(j);
                    }
                }
            }
        }
    }
    const INF: i32 = 1000000000;
    fn decompose_loops_rec(
        g: &Vec<Vec<usize>>,
        loops: &mut Vec<Vec<usize>>,
        crt: &mut Vec<usize>,
        u: usize,
        used: i32,
        mut crt_cost: i32,
        mut single_used: bool,
        best: &mut Vec<Vec<Vec<usize>>>,
        min: &mut i32,
    ) {
        if crt.len() > 0 && crt[0] == u {
            loops.push(crt.clone());
            if crt.len() > 1 {
                crt_cost += (crt.len() as i32) / 2;
            }
            if *min < crt_cost {
                loops.pop();
                return;
            }
            for s in 0..24 {
                if used >> s & 1 == 0 {
                    decompose_loops_rec(g, loops, &mut vec![], s, used, crt_cost, false, best, min);
                    loops.pop();
                    return;
                }
            }
            if loops.iter().filter(|a| a.len() % 2 == 0).count() % 2 == 0 {
                if *min > crt_cost {
                    *min = crt_cost;
                    *best = vec![];
                }
                best.push(loops.clone());
            }
            loops.pop();
            return;
        } else {
            if g[u].contains(&u) {
                if single_used {
                    return;
                }
                single_used = true;
            }
            crt.push(u);
            for &v in &g[u] {
                if used >> v & 1 == 0 {
                    decompose_loops_rec(g, loops, crt, v, used | 1 << v, crt_cost, single_used, best, min);
                }
            }
            crt.pop();
        }
    }
    let mut min = INF;
    let mut best = vec![];
    decompose_loops_rec(&g, &mut vec![], &mut vec![], 0, 0, 0, false, &mut best, &mut min);
    sum += min;
    let mut can_use = mat![false; 24; 24; 24];
    for loops in &best {
        for a in loops {
            if a.len() < 2 {
                continue;
            } else if a.len() == 2 {
                for b in loops {
                    if b.len() != 2 || a == b {
                        continue;
                    }
                    let i = a[0];
                    let j = a[1];
                    let k = b[0];
                    let l = b[1];
                    can_use[i][j][k] = true;
                    can_use[i][j][l] = true;
                    can_use[j][i][k] = true;
                    can_use[j][i][l] = true;
                }
            } else {
                for i in 0..a.len() {
                    let j = (i + 1) % a.len();
                    let k = (i + 2) % a.len();
                    can_use[a[i]][a[j]][a[k]] = true;
                }
            }
        }
    }
    // for loops in &best {
    //     for a in loops {
    //         if a.len() < 2 {
    //             continue;
    //         } else if a.len() >= 3 {
    //             for i in 0..a.len() {
    //                 let j = (i + 1) % a.len();
    //                 if a.len() % 2 != 0 {
    //                     for k in (2..a.len()).step_by(2) {
    //                         let k = (i + k) % a.len();
    //                         can_use[a[i]][a[j]][a[k]] = true;
    //                     }
    //                 } else {
    //                     for k in 2..a.len() {
    //                         let k = (i + k) % a.len();
    //                         can_use[a[i]][a[j]][a[k]] = true;
    //                     }
    //                 }
    //             }
    //         }
    //         if a.len() % 2 == 0 {
    //             for b in loops {
    //                 if b.len() % 2 != 0 || a == b {
    //                     continue;
    //                 }
    //                 for p in 0..a.len() {
    //                     for q in 0..b.len() {
    //                         let i = a[p];
    //                         let j = a[(p + 1) % a.len()];
    //                         let k = b[q];
    //                         can_use[i][j][k] = true;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    for i in 0..24 {
        for j in 0..24 {
            for k in 0..24 {
                if can_use[i][j][k] {
                    if j < i && j < k {
                        can_use[j][k][i] = true;
                        can_use[i][j][k] = false;
                    } else if k < i && k < j {
                        can_use[k][i][j] = true;
                        can_use[i][j][k] = false;
                    }
                }
            }
        }
    }
    for i in 0..24 {
        for j in 0..24 {
            for k in 0..24 {
                if can_use[i][j][k] {
                    let mut d = 0.0;
                    for a in match cluster {
                        Cluster::Edge(vs) => {
                            // TODO: 辺はペアで1色扱いにするべき？
                            vec![[vs[i].0, vs[j].0, vs[k].0], [vs[i].1, vs[j].1, vs[k].1]]
                        }
                        Cluster::Inner(vs, _) => {
                            vec![[vs[i], vs[j], vs[k]]]
                        }
                    } {
                        for i in 0..3 {
                            if start[a[i]] != input.target[a[i]] {
                                if input.color_type == ColorType::Colorful {
                                    if start[a[i]] / input.n2 as u16 + input.target[a[i]] / input.n2 as u16 == 5 {
                                        d -= OPPOSITE_WEIGHT;
                                    } else {
                                        d -= 1.0;
                                    }
                                } else {
                                    if start[a[i]] + input.target[a[i]] == 5 {
                                        d -= OPPOSITE_WEIGHT;
                                    } else {
                                        d -= 1.0;
                                    }
                                }
                            }
                            if start[a[i]] != input.target[a[(i + 1) % 3]] {
                                if input.color_type == ColorType::Colorful {
                                    if start[a[i]] / input.n2 as u16 + input.target[a[(i + 1) % 3]] / input.n2 as u16 == 5 {
                                        d += OPPOSITE_WEIGHT;
                                    } else {
                                        d += 1.0;
                                    }
                                } else {
                                    if start[a[i]] + input.target[a[(i + 1) % 3]] == 5 {
                                        d += OPPOSITE_WEIGHT;
                                    } else {
                                        d += 1.0;
                                    }
                                }
                            }
                        }
                    }
                    list.push((i, j, k, d));
                }
            }
        }
    }
    (list, sum)
}

fn decompose_clusters(input: &Input) -> (Vec<Vec<(usize, usize, usize, f64)>>, i32) {
    let mut list = vec![vec![]; input.clusters.len()];
    let mut sum = 0;
    for c in 0..input.clusters.len() {
        let (l, s) = decompose_cluster(input, &input.start, c);
        list[c] = l;
        sum += s;
    }
    (list, sum)
}

// 直前の操作の末尾と追加する操作の先頭を打ち消してからつなげる
fn extend_with_overlap(out: &mut Vec<usize>, add: &[usize], n: usize) {
    if add.len() == 0 {
        return;
    } else if out.len() == 0 {
        out.extend(add.iter().cloned());
        return;
    }
    let mut s = out.len() - 1;
    while s >= 1 && out[s - 1] / n % 3 == out[s] / n % 3 {
        s -= 1;
    }
    let mut t = 0;
    while t + 1 < add.len() && add[t] / n % 3 == add[t + 1] / n % 3 {
        t += 1;
    }
    let mut ok = true;
    for i in 0..=t {
        if let Some(p) = out[s..].iter().position(|&k| k == (add[i] + 3 * n) % (6 * n)) {
            out.remove(s + p);
            ok = false;
        } else {
            out.push(add[i]);
        }
    }
    if ok {
        out.extend(add[t + 1..].iter().cloned());
    } else {
        extend_with_overlap(out, &add[t + 1..], n);
    }
}

fn overlap(out: &[usize], add: &[usize], n: usize) -> i32 {
    if add.len() == 0 || out.len() == 0 {
        return 0;
    }
    let mut s = out.len() - 1;
    while s >= 1 && out[s - 1] / n % 3 == out[s] / n % 3 {
        s -= 1;
    }
    let mut t = 0;
    while t + 1 < add.len() && add[t] / n % 3 == add[t + 1] / n % 3 {
        t += 1;
    }
    let mut ov = 0;
    let mut tmp = out[s..].to_vec();
    let mut ok = true;
    for i in 0..=t {
        if let Some(p) = tmp.iter().position(|&k| k == (add[i] + 3 * n) % (6 * n)) {
            tmp.remove(p);
            ov += 2;
        } else {
            ok = false;
        }
    }
    if tmp.len() > 0 || !ok {
        ov
    } else {
        ov + overlap(&out[..s], &add[t + 1..], n)
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
        loops.push(tmp);
    }
    loops
}

const OPPOSITE_WEIGHT: f64 = 1.0;

fn diff2(input: &Input, start: &[u16]) -> f64 {
    let mut sum = 0.0;
    if input.color_type == ColorType::Colorful {
        let n2 = start.len() as u16 / 6;
        for (&a, &b) in start.iter().zip(&input.target) {
            if a != b {
                if a / n2 + b / n2 == 5 {
                    sum += OPPOSITE_WEIGHT;
                } else {
                    sum += 1.0;
                }
            }
        }
    } else {
        for (&a, &b) in start.iter().zip(&input.target) {
            if a != b {
                if a + b == 5 {
                    sum += OPPOSITE_WEIGHT;
                } else {
                    sum += 1.0;
                }
            }
        }
    }
    sum
}

fn apply_fast<T: Copy>(input: &Input, crt: &mut Vec<T>, mv: usize) {
    for a in &input.rotn[mv] {
        let s = crt[a[0]];
        for i in 0..a.len() - 1 {
            crt[a[i]] = crt[a[i + 1]];
        }
        crt[a[a.len() - 1]] = s;
    }
}

fn apply_loop(input: &Input, crt: &mut Vec<u16>, mv: usize) -> f64 {
    if input.color_type == ColorType::Colorful {
        return apply_loop_colorful(input, crt, mv);
    }
    let mut diff = 0.0;
    for a in &input.rotn[mv] {
        let s = crt[a[0]];
        for i in 0..a.len() - 1 {
            if crt[a[i]] != input.target[a[i]] {
                if crt[a[i]] + input.target[a[i]] == 5 {
                    diff -= OPPOSITE_WEIGHT;
                } else {
                    diff -= 1.0;
                }
            }
            crt[a[i]] = crt[a[i + 1]];
            if crt[a[i]] != input.target[a[i]] {
                if crt[a[i]] + input.target[a[i]] == 5 {
                    diff += OPPOSITE_WEIGHT;
                } else {
                    diff += 1.0;
                }
            }
        }
        if crt[a[a.len() - 1]] != input.target[a[a.len() - 1]] {
            if crt[a[a.len() - 1]] + input.target[a[a.len() - 1]] == 5 {
                diff -= OPPOSITE_WEIGHT;
            } else {
                diff -= 1.0;
            }
        }
        crt[a[a.len() - 1]] = s;
        if crt[a[a.len() - 1]] != input.target[a[a.len() - 1]] {
            if crt[a[a.len() - 1]] + input.target[a[a.len() - 1]] == 5 {
                diff += OPPOSITE_WEIGHT;
            } else {
                diff += 1.0;
            }
        }
    }
    diff
}

fn apply_loop_colorful(input: &Input, crt: &mut Vec<u16>, mv: usize) -> f64 {
    let n2 = crt.len() as u16 / 6;
    let mut diff = 0.0;
    for a in &input.rotn[mv] {
        let s = crt[a[0]];
        for i in 0..a.len() - 1 {
            if crt[a[i]] != input.target[a[i]] {
                if crt[a[i]] / n2 + input.target[a[i]] / n2 == 5 {
                    diff -= OPPOSITE_WEIGHT;
                } else {
                    diff -= 1.0;
                }
            }
            crt[a[i]] = crt[a[i + 1]];
            if crt[a[i]] != input.target[a[i]] {
                if crt[a[i]] / n2 + input.target[a[i]] / n2 == 5 {
                    diff += OPPOSITE_WEIGHT;
                } else {
                    diff += 1.0;
                }
            }
        }
        if crt[a[a.len() - 1]] != input.target[a[a.len() - 1]] {
            if crt[a[a.len() - 1]] / n2 + input.target[a[a.len() - 1]] / n2 == 5 {
                diff -= OPPOSITE_WEIGHT;
            } else {
                diff -= 1.0;
            }
        }
        crt[a[a.len() - 1]] = s;
        if crt[a[a.len() - 1]] != input.target[a[a.len() - 1]] {
            if crt[a[a.len() - 1]] / n2 + input.target[a[a.len() - 1]] / n2 == 5 {
                diff += OPPOSITE_WEIGHT;
            } else {
                diff += 1.0;
            }
        }
    }
    diff
}

fn apply_loops(input: &Input, crt: &mut Vec<u16>, mv: &[usize]) -> f64 {
    let mut diff = 0.0;
    for &k in mv {
        diff += apply_loop(input, crt, k);
    }
    diff
}

fn get_dst(cluster: &Cluster, crt: &Vec<u16>, target: &Vec<u16>) -> Vec<usize> {
    match cluster {
        Cluster::Edge(vs) => {
            let mut dst = vec![!0; vs.len()];
            let mut used = vec![false; vs.len()];
            for i in 0..vs.len() {
                let (v1, v2) = vs[i];
                if crt[v1] == target[v1] && crt[v2] == target[v2] {
                    dst[i] = i;
                    used[i] = true;
                }
            }
            for i in 0..vs.len() {
                if dst[i] == !0 {
                    let (v1, v2) = vs[i];
                    let j = (0..vs.len())
                        .find(|&j| !used[j] && crt[v1] == target[vs[j].0] && crt[v2] == target[vs[j].1])
                        .unwrap();
                    dst[i] = j;
                    used[j] = true;
                }
            }
            dst
        }
        Cluster::Inner(vs, _) => {
            let mut dst = vec![!0; vs.len()];
            let mut used = vec![false; vs.len()];
            for i in 0..vs.len() {
                let v = vs[i];
                if crt[v] == target[v] {
                    dst[i] = i;
                    used[i] = true;
                }
            }
            for i in 0..vs.len() {
                if dst[i] == !0 {
                    let v = vs[i];
                    let j = (0..vs.len()).find(|&j| !used[j] && crt[v] == target[vs[j]]).unwrap();
                    dst[i] = j;
                    used[j] = true;
                }
            }
            dst
        }
    }
}

fn is_odd(perm: &Vec<usize>) -> bool {
    let mut used = vec![false; perm.len()];
    let mut odd = false;
    for i in 0..perm.len() {
        if used[i] {
            continue;
        }
        let mut len = 0;
        let mut v = i;
        while used[v].setmax(true) {
            len += 1;
            v = perm[v];
        }
        if len % 2 == 0 {
            odd = !odd;
        }
    }
    odd
}

const ROT_X: [usize; 24] = [
    1, 3, 0, 2, 16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 20, 23, 21,
];

const ROT_Y: [usize; 24] = [
    18, 16, 19, 17, 6, 4, 7, 5, 2, 0, 3, 1, 13, 15, 12, 14, 22, 20, 23, 21, 10, 8, 11, 9,
];

const ROT_Z: [usize; 24] = [
    4, 5, 6, 7, 20, 21, 22, 23, 10, 8, 11, 9, 3, 2, 1, 0, 17, 19, 16, 18, 15, 14, 13, 12,
];

#[derive(Clone, Debug)]
struct Rot3 {
    n: usize,
    data: Vec<Vec<Vec<Vec<Vec<Vec<usize>>>>>>,
}

impl Rot3 {
    fn new(n: usize) -> Self {
        let mut data = mat![vec![]; 4; 24; 24; 24];
        let mut lines = include_str!("../rotate.txt").lines();
        for iter in 0..4 {
            assert!(lines.next().unwrap().starts_with("#"));
            let m = [4, 5, 6, 4][iter];
            for _ in 0..23 * 22 / 2 {
                let ss = lines.next().unwrap().split_whitespace().collect_vec();
                assert_eq!(ss[0], "0");
                let i = ss[1].parse::<usize>().unwrap();
                let j = ss[2].parse::<usize>().unwrap();
                let mv = if iter == 0 || iter == 3 {
                    ss[3]
                        .split('.')
                        .map(|s| {
                            let (s, add) = if s.starts_with('-') { (&s[1..], 12) } else { (s, 0) };
                            add + ["d0", "d1", "d2", "d3", "f0", "f1", "f2", "f3", "r0", "r1", "r2", "r3"]
                                .iter()
                                .position(|&x| x == s)
                                .unwrap()
                        })
                        .collect_vec()
                } else if iter == 1 {
                    ss[3]
                        .split('.')
                        .map(|s| {
                            let (s, add) = if s.starts_with('-') { (&s[1..], 15) } else { (s, 0) };
                            add + [
                                "d0", "d1", "d2", "d3", "d4", "f0", "f1", "f2", "f3", "f4", "r0", "r1", "r2", "r3", "r4",
                            ]
                            .iter()
                            .position(|&x| x == s)
                            .unwrap()
                        })
                        .collect_vec()
                } else {
                    ss[3]
                        .split('.')
                        .map(|s| {
                            let (s, add) = if s.starts_with('-') { (&s[1..], 18) } else { (s, 0) };
                            add + [
                                "d0", "d1", "d2", "d3", "d4", "d5", "f0", "f1", "f2", "f3", "f4", "f5", "r0", "r1", "r2", "r3",
                                "r4", "r5",
                            ]
                            .iter()
                            .position(|&x| x == s)
                            .unwrap()
                        })
                        .collect_vec()
                };
                let (i, j) = if iter == 1 {
                    (
                        if i % 4 == 1 {
                            i + 1
                        } else if i % 4 == 2 {
                            i - 1
                        } else {
                            i
                        },
                        if j % 4 == 1 {
                            j + 1
                        } else if j % 4 == 2 {
                            j - 1
                        } else {
                            j
                        },
                    )
                } else {
                    (i, j)
                };
                data[iter][0][i][j] = vec![mv.clone()];
                data[iter][0][j][i] = vec![mv.into_iter().map(|x| (x + 3 * m) % (6 * m)).rev().collect_vec()];
            }
        }
        let mut lines = include_str!("../rotate_all.txt").lines();
        assert!(lines.next().unwrap().starts_with("#"));
        for iter in 0..4 {
            let m = [4, 5, 6, 4][iter];
            loop {
                let ss = lines.next().unwrap().split_whitespace().collect_vec();
                if ss[0] == "#" {
                    break;
                }
                assert_eq!(ss[0], "0");
                let i = ss[1].parse::<usize>().unwrap();
                let j = ss[2].parse::<usize>().unwrap();
                let mut tmp = vec![];
                for a in 3..ss.len() {
                    let mv = if iter == 0 || iter == 3 {
                        ss[a]
                            .split('.')
                            .map(|s| {
                                let (s, add) = if s.starts_with('-') { (&s[1..], 12) } else { (s, 0) };
                                add + ["d0", "d1", "d2", "d3", "f0", "f1", "f2", "f3", "r0", "r1", "r2", "r3"]
                                    .iter()
                                    .position(|&x| x == s)
                                    .unwrap()
                            })
                            .collect_vec()
                    } else if iter == 1 {
                        ss[a]
                            .split('.')
                            .map(|s| {
                                let (s, add) = if s.starts_with('-') { (&s[1..], 15) } else { (s, 0) };
                                add + [
                                    "d0", "d1", "d2", "d3", "d4", "f0", "f1", "f2", "f3", "f4", "r0", "r1", "r2", "r3", "r4",
                                ]
                                .iter()
                                .position(|&x| x == s)
                                .unwrap()
                            })
                            .collect_vec()
                    } else {
                        ss[a]
                            .split('.')
                            .map(|s| {
                                let (s, add) = if s.starts_with('-') { (&s[1..], 18) } else { (s, 0) };
                                add + [
                                    "d0", "d1", "d2", "d3", "d4", "d5", "f0", "f1", "f2", "f3", "f4", "f5", "r0", "r1", "r2",
                                    "r3", "r4", "r5",
                                ]
                                .iter()
                                .position(|&x| x == s)
                                .unwrap()
                            })
                            .collect_vec()
                    };
                    tmp.push(mv);
                }
                let (i, j) = if iter == 1 {
                    (
                        if i % 4 == 1 {
                            i + 1
                        } else if i % 4 == 2 {
                            i - 1
                        } else {
                            i
                        },
                        if j % 4 == 1 {
                            j + 1
                        } else if j % 4 == 2 {
                            j - 1
                        } else {
                            j
                        },
                    )
                } else {
                    (i, j)
                };
                data[iter][0][i][j] = tmp.clone();
                data[iter][0][j][i] = tmp
                    .iter()
                    .map(|mv| mv.into_iter().map(|x| (x + 3 * m) % (6 * m)).rev().collect_vec())
                    .collect_vec();
            }
        }
        for iter in 0..4 {
            let m = [4, 5, 6, 4][iter];
            let mut ids = (0..24).collect_vec();
            let mut moves = (0..6 * m).collect_vec();
            for r in 0..6 {
                for _ in 0..4 {
                    for i in 0..24 {
                        for j in 0..24 {
                            data[iter][ids[0]][ids[i]][ids[j]] = data[iter][0][i][j]
                                .iter()
                                .map(|a| a.iter().map(|&k| moves[k]).collect_vec())
                                .collect_vec();
                        }
                    }
                    ids = apply(&ids, &ROT_Z);
                    moves = (4 * m..5 * m)
                        .rev()
                        .chain(0 * m..1 * m)
                        .chain(2 * m..3 * m)
                        .chain((1 * m..2 * m).rev())
                        .chain(3 * m..4 * m)
                        .chain(5 * m..6 * m)
                        .map(|i| moves[i])
                        .collect_vec();
                }
                if r % 2 == 0 {
                    ids = apply(&ids, &ROT_Y);
                    moves = (2 * m..3 * m)
                        .chain(1 * m..2 * m)
                        .chain((3 * m..4 * m).rev())
                        .chain(5 * m..6 * m)
                        .chain(4 * m..5 * m)
                        .chain((0 * m..1 * m).rev())
                        .map(|i| moves[i])
                        .collect_vec();
                } else {
                    ids = apply(&ids, &ROT_X);
                    moves = (0 * m..1 * m)
                        .chain((5 * m..6 * m).rev())
                        .chain(1 * m..2 * m)
                        .chain(3 * m..4 * m)
                        .chain((2 * m..3 * m).rev())
                        .chain(4 * m..5 * m)
                        .map(|i| moves[i])
                        .collect_vec();
                }
            }
        }
        Self { n, data }
    }
    fn get(&self, cluster: &Cluster, i: usize, j: usize, k: usize) -> Vec<Vec<usize>> {
        match cluster {
            Cluster::Edge(vs) => {
                let d = vs[0].0 % self.n;
                self.data[3][i][j][k]
                    .iter()
                    .map(|a| {
                        a.iter()
                            .map(|&x| {
                                x / 4 * self.n
                                    + match x % 4 {
                                        0 => 0,
                                        1 => d,
                                        2 => self.n - 1 - d,
                                        3 => self.n - 1,
                                        _ => unreachable!(),
                                    }
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            }
            Cluster::Inner(vs, rotate_type) => {
                if *rotate_type == 0 {
                    let d = vs[0] / self.n;
                    self.data[0][i][j][k]
                        .iter()
                        .map(|a| {
                            a.iter()
                                .map(|&x| {
                                    x / 4 * self.n
                                        + match x % 4 {
                                            0 => 0,
                                            1 => d,
                                            2 => self.n - 1 - d,
                                            3 => self.n - 1,
                                            _ => unreachable!(),
                                        }
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                } else if *rotate_type == 1 {
                    let d = vs[0] / self.n;
                    self.data[1][i][j][k]
                        .iter()
                        .map(|a| {
                            a.iter()
                                .map(|&x| {
                                    x / 5 * self.n
                                        + match x % 5 {
                                            0 => 0,
                                            1 => d,
                                            2 => self.n / 2,
                                            3 => self.n - 1 - d,
                                            4 => self.n - 1,
                                            _ => unreachable!(),
                                        }
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                } else {
                    let d = vs[0] / self.n;
                    let e = vs[0] % self.n;
                    self.data[2][i][j][k]
                        .iter()
                        .map(|a| {
                            a.iter()
                                .map(|&x| {
                                    x / 6 * self.n
                                        + match x % 6 {
                                            0 => 0,
                                            1 => d,
                                            2 => e,
                                            3 => self.n - 1 - e,
                                            4 => self.n - 1 - d,
                                            5 => self.n - 1,
                                            _ => unreachable!(),
                                        }
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                }
            }
        }
    }
}

fn gen_clusters(n: usize) -> Vec<Cluster> {
    macro_rules! id {
        ($p:expr, $i:expr, $j:expr) => {
            ($p * n * n + $i * n + $j)
        };
    }
    let mut clusters = vec![];
    for i in 1..n / 2 {
        clusters.push(Cluster::Inner(
            vec![
                id!(0, i, i),
                id!(0, i, n - 1 - i),
                id!(0, n - 1 - i, i),
                id!(0, n - 1 - i, n - 1 - i),
                id!(1, i, i),
                id!(1, i, n - 1 - i),
                id!(1, n - 1 - i, i),
                id!(1, n - 1 - i, n - 1 - i),
                id!(2, i, i),
                id!(2, i, n - 1 - i),
                id!(2, n - 1 - i, i),
                id!(2, n - 1 - i, n - 1 - i),
                id!(3, i, i),
                id!(3, i, n - 1 - i),
                id!(3, n - 1 - i, i),
                id!(3, n - 1 - i, n - 1 - i),
                id!(4, i, i),
                id!(4, i, n - 1 - i),
                id!(4, n - 1 - i, i),
                id!(4, n - 1 - i, n - 1 - i),
                id!(5, i, i),
                id!(5, i, n - 1 - i),
                id!(5, n - 1 - i, i),
                id!(5, n - 1 - i, n - 1 - i),
            ],
            0,
        ));
    }
    if n % 2 != 0 {
        for i in 1..n / 2 {
            clusters.push(Cluster::Inner(
                vec![
                    id!(0, i, n / 2),
                    id!(0, n / 2, n - 1 - i),
                    id!(0, n / 2, i),
                    id!(0, n - 1 - i, n / 2),
                    id!(1, i, n / 2),
                    id!(1, n / 2, n - 1 - i),
                    id!(1, n / 2, i),
                    id!(1, n - 1 - i, n / 2),
                    id!(2, i, n / 2),
                    id!(2, n / 2, n - 1 - i),
                    id!(2, n / 2, i),
                    id!(2, n - 1 - i, n / 2),
                    id!(3, i, n / 2),
                    id!(3, n / 2, n - 1 - i),
                    id!(3, n / 2, i),
                    id!(3, n - 1 - i, n / 2),
                    id!(4, i, n / 2),
                    id!(4, n / 2, n - 1 - i),
                    id!(4, n / 2, i),
                    id!(4, n - 1 - i, n / 2),
                    id!(5, i, n / 2),
                    id!(5, n / 2, n - 1 - i),
                    id!(5, n / 2, i),
                    id!(5, n - 1 - i, n / 2),
                ],
                1,
            ));
        }
    }
    for x in 1..n / 2 {
        for y in x + 1..n - 1 {
            if y * 2 != n - 1 && y < n - 1 - x {
                clusters.push(Cluster::Inner(
                    vec![
                        id!(0, x, y),
                        id!(0, y, n - 1 - x),
                        id!(0, n - 1 - y, x),
                        id!(0, n - 1 - x, n - 1 - y),
                        id!(1, x, y),
                        id!(1, y, n - 1 - x),
                        id!(1, n - 1 - y, x),
                        id!(1, n - 1 - x, n - 1 - y),
                        id!(2, x, y),
                        id!(2, y, n - 1 - x),
                        id!(2, n - 1 - y, x),
                        id!(2, n - 1 - x, n - 1 - y),
                        id!(3, x, y),
                        id!(3, y, n - 1 - x),
                        id!(3, n - 1 - y, x),
                        id!(3, n - 1 - x, n - 1 - y),
                        id!(4, x, y),
                        id!(4, y, n - 1 - x),
                        id!(4, n - 1 - y, x),
                        id!(4, n - 1 - x, n - 1 - y),
                        id!(5, x, y),
                        id!(5, y, n - 1 - x),
                        id!(5, n - 1 - y, x),
                        id!(5, n - 1 - x, n - 1 - y),
                    ],
                    2,
                ));
            }
        }
    }
    for i in 1..n / 2 {
        clusters.push(Cluster::Edge(vec![
            (id!(0, 0, i), id!(3, 0, n - 1 - i)),
            (id!(0, i, n - 1), id!(2, 0, n - 1 - i)),
            (id!(0, n - 1 - i, 0), id!(4, 0, n - 1 - i)),
            (id!(0, n - 1, n - 1 - i), id!(1, 0, n - 1 - i)),
            (id!(1, 0, i), id!(0, n - 1, i)),
            (id!(1, i, n - 1), id!(2, i, 0)),
            (id!(1, n - 1 - i, 0), id!(4, n - 1 - i, n - 1)),
            (id!(1, n - 1, n - 1 - i), id!(5, 0, n - 1 - i)),
            (id!(2, 0, i), id!(0, n - 1 - i, n - 1)),
            (id!(2, i, n - 1), id!(3, i, 0)),
            (id!(2, n - 1 - i, 0), id!(1, n - 1 - i, n - 1)),
            (id!(2, n - 1, n - 1 - i), id!(5, n - 1 - i, n - 1)),
            (id!(3, 0, i), id!(0, 0, n - 1 - i)),
            (id!(3, i, n - 1), id!(4, i, 0)),
            (id!(3, n - 1 - i, 0), id!(2, n - 1 - i, n - 1)),
            (id!(3, n - 1, n - 1 - i), id!(5, n - 1, i)),
            (id!(4, 0, i), id!(0, i, 0)),
            (id!(4, i, n - 1), id!(1, i, 0)),
            (id!(4, n - 1 - i, 0), id!(3, n - 1 - i, n - 1)),
            (id!(4, n - 1, n - 1 - i), id!(5, i, 0)),
            (id!(5, 0, i), id!(1, n - 1, i)),
            (id!(5, i, n - 1), id!(2, n - 1, i)),
            (id!(5, n - 1 - i, 0), id!(4, n - 1, i)),
            (id!(5, n - 1, n - 1 - i), id!(3, n - 1, i)),
        ]));
    }
    clusters
}

#[derive(Clone, Debug)]
enum Cluster {
    Edge(Vec<(usize, usize)>),
    /// 0: 対角、1: 十字、2: その他
    Inner(Vec<usize>, usize),
}

pub fn vis(ps: &Vec<String>, n: usize) {
    for i in 0..n {
        for _ in 0..n {
            eprint!("{:4}", "");
        }
        for j in 0..n {
            eprint!("{:4}", ps[i * n + j]);
        }
        eprintln!();
    }
    for i in 0..n {
        for j in 0..n {
            eprint!("{:4}", ps[4 * n * n + i * n + j]);
        }
        for j in 0..n {
            eprint!("{:4}", ps[1 * n * n + i * n + j]);
        }
        for j in 0..n {
            eprint!("{:4}", ps[2 * n * n + i * n + j]);
        }
        for j in 0..n {
            eprint!("{:4}", ps[3 * n * n + i * n + j]);
        }
        eprintln!();
    }
    for i in 0..n {
        for _ in 0..n {
            eprint!("{:4}", "");
        }
        for j in 0..n {
            eprint!("{:4}", ps[5 * n * n + i * n + j]);
        }
        eprintln!();
    }
}

const MOVES2: [[usize; 24]; 12] = [
    [
        0, 1, 2, 3, 4, 5, 18, 19, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 22, 20, 23, 21,
    ],
    [
        1, 3, 0, 2, 16, 17, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 20, 21, 22, 23,
    ],
    [
        0, 1, 19, 17, 6, 4, 7, 5, 2, 9, 3, 11, 12, 13, 14, 15, 16, 20, 18, 21, 10, 8, 22, 23,
    ],
    [
        18, 16, 2, 3, 4, 5, 6, 7, 8, 0, 10, 1, 13, 15, 12, 14, 22, 17, 23, 19, 20, 21, 11, 9,
    ],
    [
        0, 5, 2, 7, 4, 21, 6, 23, 10, 8, 11, 9, 3, 13, 1, 15, 16, 17, 18, 19, 20, 14, 22, 12,
    ],
    [
        4, 1, 6, 3, 20, 5, 22, 7, 8, 9, 10, 11, 12, 2, 14, 0, 17, 19, 16, 18, 15, 21, 13, 23,
    ],
    [
        0, 1, 2, 3, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 6, 7, 21, 23, 20, 22,
    ],
    [
        2, 0, 3, 1, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 4, 5, 18, 19, 20, 21, 22, 23,
    ],
    [
        0, 1, 8, 10, 5, 7, 4, 6, 21, 9, 20, 11, 12, 13, 14, 15, 16, 3, 18, 2, 17, 19, 22, 23,
    ],
    [
        9, 11, 2, 3, 4, 5, 6, 7, 8, 23, 10, 22, 14, 12, 15, 13, 1, 17, 0, 19, 20, 21, 16, 18,
    ],
    [
        0, 14, 2, 12, 4, 1, 6, 3, 9, 11, 8, 10, 23, 13, 21, 15, 16, 17, 18, 19, 20, 5, 22, 7,
    ],
    [
        15, 1, 13, 3, 0, 5, 2, 7, 8, 9, 10, 11, 12, 22, 14, 20, 18, 16, 19, 17, 4, 21, 6, 23,
    ],
];

/// 1x1x1を解く
fn solve1(crt: &Vec<u16>) -> Vec<usize> {
    solve_bfs(
        crt,
        &(0..6).collect_vec(),
        &vec![
            vec![0, 4, 1, 2, 3, 5],
            vec![4, 1, 0, 3, 5, 2],
            vec![1, 5, 2, 0, 4, 3],
            inv_perm(&vec![0, 4, 1, 2, 3, 5]),
            inv_perm(&vec![4, 1, 0, 3, 5, 2]),
            inv_perm(&vec![1, 5, 2, 0, 4, 3]),
        ],
    )
}

/// 両側探索で2x2x2以下を解く
fn solve_bfs(crt: &Vec<u16>, target: &Vec<u16>, moves: &Vec<Vec<usize>>) -> Vec<usize> {
    let mut trace = Trace::new();
    let mut fwd = HashMap::new();
    let mut bwd = HashMap::new();
    let mut que = VecDeque::new();
    que.push_back((crt.clone(), !0, 0));
    que.push_back((target.clone(), !0, 1));
    fwd.insert(crt.clone(), !0);
    bwd.insert(target.clone(), !0);
    while let Some((crt, id, f)) = que.pop_front() {
        if f == 0 {
            if let Some(b) = bwd.get(&crt) {
                return trace.get(id).into_iter().chain(trace.get(*b).into_iter().rev()).collect_vec();
            }
            for mv in 0..moves.len() {
                let next = apply(&crt, &moves[mv]);
                if !fwd.contains_key(&next) {
                    let id = trace.add(mv, id);
                    fwd.insert(next.clone(), id);
                    que.push_back((next, id, 0));
                }
            }
        } else {
            if let Some(f) = fwd.get(&crt) {
                return trace.get(*f).into_iter().chain(trace.get(id).into_iter().rev()).collect_vec();
            }
            for mv in 0..moves.len() {
                let next = apply(&crt, &moves[mv]);
                if !bwd.contains_key(&next) {
                    let id = trace.add((mv + moves.len() / 2) % moves.len(), id);
                    bwd.insert(next.clone(), id);
                    que.push_back((next, id, 1));
                }
            }
        }
    }
    panic!();
}

/// 既存のソルバに投げて解く
fn solve_small(crt: &Vec<u16>, n: usize) -> Vec<usize> {
    let m = 3 * n;
    let mut out = vec![];
    let input2 = crt[0..n * n]
        .iter()
        .chain(&crt[2 * n * n..3 * n * n])
        .chain(&crt[n * n..2 * n * n])
        .chain(&crt[5 * n * n..])
        .chain(&crt[4 * n * n..5 * n * n])
        .chain(&crt[3 * n * n..4 * n * n])
        .map(|&i| ["U", "F", "R", "B", "L", "D"][i as usize])
        .join("");
    let mut iter = 0;
    let out2 = loop {
        if let Ok(out2) = cmd!("../ext/rubiks-cube-NxNxN-solver/rubiks-cube-solver.py", "--state", &input2)
            .dir("../ext/rubiks-cube-NxNxN-solver/")
            .stderr_null()
            .read()
        {
            break out2
                .trim_start_matches("Solution: ")
                .trim()
                .split(' ')
                .map(|s| s.to_owned())
                .collect_vec();
        }
        iter += 1;
        if iter >= 10 {
            panic!();
        }
    };
    for mut mv in out2 {
        let mut tmp = vec![];
        let w = if mv.contains("w") {
            if mv.chars().next().unwrap().is_ascii_digit() {
                let mut p = 0;
                while mv.chars().nth(p + 1).unwrap().is_ascii_digit() {
                    p += 1;
                }
                mv[..=p].parse::<usize>().unwrap()
            } else {
                2
            }
        } else {
            1
        };
        for i in 0..w {
            if mv.contains("D") {
                tmp.push(i);
            } else if mv.contains("U") {
                tmp.push(n - 1 - i + m);
            } else if mv.contains("F") {
                tmp.push(n + i);
            } else if mv.contains("B") {
                tmp.push(2 * n - 1 - i + m);
            } else if mv.contains("R") {
                tmp.push(2 * n + i);
            } else if mv.contains("L") {
                tmp.push(3 * n - 1 - i + m);
            } else {
                panic!("unknown mv: {}", mv);
            }
        }
        if mv.ends_with("'") {
            for v in &mut tmp {
                *v = (*v + m) % (m * 2);
            }
            mv = mv.trim_end_matches("'").to_owned();
        }
        let rep = if mv.chars().last().unwrap().is_ascii_digit() {
            let mut p = mv.len() - 1;
            while mv.chars().nth(p - 1).unwrap().is_ascii_digit() {
                p -= 1;
            }
            mv[p..].parse::<usize>().unwrap()
        } else {
            1
        };
        for _ in 0..rep {
            out.extend(tmp.clone());
        }
    }
    // 向きが違う場合があるので正しい向きに直す
    let mut crt2 = vec![];
    for k in 0..6 {
        for i in 0..2 {
            for j in 0..2 {
                crt2.push(crt[k * n * n + (i * (n - 1)) * n + j * (n - 1)]);
            }
        }
    }
    for &mv in &out {
        if mv % n == 0 {
            crt2 = apply(&crt2, &MOVES2[mv / n * 2]);
        } else if mv % n == n - 1 {
            crt2 = apply(&crt2, &MOVES2[mv / n * 2 + 1]);
        }
    }
    let out1 = solve1(&crt2.iter().cloned().step_by(4).collect_vec());
    for mv in out1 {
        for k in 0..n {
            out.push(mv * n + k);
        }
    }
    out
}
