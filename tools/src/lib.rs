#![allow(non_snake_case, unused_macros)]

use itertools::Itertools;
use once_cell::sync::Lazy;
use proconio::input;
use std::{collections::HashMap, f64::consts::PI};
use svg::node::{
    element::{Circle, Group, Rectangle, Style, Title},
    Text,
};

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
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

#[derive(Clone, Debug)]
pub struct Input {
    pub id: usize,
    pub puzzle_type: String,
    pub n: usize,
    pub m: usize,
    pub moves: Vec<Vec<usize>>,
    pub move_names: Vec<String>,
    pub target: Vec<u16>,
    pub start: Vec<u16>,
    pub wildcard: usize,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{} {} {} {} {}",
            self.id,
            self.puzzle_type,
            self.n,
            self.moves.len(),
            self.wildcard
        )?;
        for i in 0..self.m {
            writeln!(f, "{} {}", self.move_names[i], self.moves[i].iter().join(" "))?;
        }
        writeln!(f, "{}", self.target.iter().join(" "))?;
        writeln!(f, "{}", self.start.iter().join(" "))?;
        Ok(())
    }
}

pub fn inv_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for i in 0..perm.len() {
        inv[perm[i]] = i;
    }
    inv
}

pub fn parse_input(f: &str) -> Input {
    let f = proconio::source::once::OnceSource::from(f);
    input! {
        from f,
        id: usize, puzzle_type: String, n: usize, num_moves: usize, wildcard: usize,
        moves: [(String, [usize; n]); num_moves],
        target: [u16; n],
        start: [u16; n],
    }
    Input {
        id,
        puzzle_type,
        n,
        m: num_moves,
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

pub struct Output {
    pub out: Vec<usize>,
}

pub fn parse_output(input: &Input, f: &str) -> Result<Output, String> {
    let ids = (0..input.move_names.len())
        .map(|i| (input.move_names[i].clone(), i))
        .collect::<HashMap<_, _>>();
    let mut out = vec![];
    let f = f.trim().lines().filter(|s| s.len() > 0).last().unwrap_or("").trim();
    if f.len() > 0 {
        for i in f.trim().split('.') {
            if let Some(&id) = ids.get(i) {
                out.push(id);
            } else {
                return Err(format!("Invalid move: {}", i));
            }
        }
    }
    Ok(Output { out })
}

static PUZZLE_INFO: Lazy<HashMap<String, Vec<(String, Vec<usize>)>>> = Lazy::new(|| {
    let csv_data = include_str!("../puzzle_info.csv");
    let mut puzzles = HashMap::new();
    for line in csv_data.lines().skip(1) {
        let line = line.trim();
        if line.len() == 0 {
            continue;
        }
        let p = line.find(',').unwrap();
        let name = line[..p].to_owned();
        let json = line[p + 2..line.len() - 1].to_owned().replace("'", "\"");
        let moves: HashMap<String, Vec<usize>> = serde_json::from_str(&json).unwrap();
        puzzles.insert(
            name,
            moves
                .into_iter()
                .sorted_by_key(|(s, _)| {
                    let mut p = s.len();
                    while s.chars().nth(p - 1).unwrap().is_ascii_digit() {
                        p -= 1;
                    }
                    (s[..p].to_owned(), s[p..].parse::<usize>().unwrap_or(0))
                })
                .collect_vec(),
        );
    }
    puzzles
});

static PUZZLES: Lazy<Vec<(String, Vec<String>, Vec<String>, usize)>> = Lazy::new(|| {
    let csv_data = include_str!("../puzzles.csv");
    let mut puzzles = vec![];
    for line in csv_data.lines().skip(1) {
        let line = line.trim();
        if line.len() == 0 {
            continue;
        }
        let ss = line.split(',').collect_vec();
        puzzles.push((
            ss[1].to_owned(),
            ss[2].split(';').map(|s| s.to_owned()).collect_vec(),
            ss[3].split(';').map(|s| s.to_owned()).collect_vec(),
            ss[4].parse::<usize>().unwrap(),
        ));
    }
    puzzles
});

pub fn gen(seed: u64) -> Input {
    let puzzle = &PUZZLES[seed as usize];
    let mut ids = HashMap::new();
    for i in 0..puzzle.1.len() {
        if !ids.contains_key(&puzzle.1[i]) {
            ids.insert(puzzle.1[i].to_owned(), ids.len() as u16);
        }
    }
    Input {
        id: seed as usize,
        puzzle_type: puzzle.0.clone(),
        n: puzzle.1.len(),
        m: PUZZLE_INFO[&puzzle.0].len(),
        moves: PUZZLE_INFO[&puzzle.0].iter().map(|(_, v)| v.clone()).collect(),
        move_names: PUZZLE_INFO[&puzzle.0].iter().map(|(v, _)| v.clone()).collect(),
        target: puzzle.1.iter().map(|s| ids[s]).collect(),
        start: puzzle.2.iter().map(|s| ids[s]).collect(),
        wildcard: puzzle.3,
    }
}

pub fn compute_score(input: &Input, out: &Output) -> (i64, String) {
    let (mut score, err, _) = compute_score_details(input, &out.out);
    if err.len() > 0 {
        score = 0;
    }
    (score, err)
}

pub fn compute_score_details(input: &Input, out: &[usize]) -> (i64, String, Vec<usize>) {
    let mut crt = (0..input.n).collect_vec();
    for &i in out {
        crt = input.moves[i].iter().map(|&p| crt[p]).collect();
    }
    let mut diff = 0;
    for i in 0..input.n {
        if input.start[crt[i]] != input.target[i] {
            diff += 1;
        }
    }
    if diff <= input.wildcard {
        (out.len() as i64, String::new(), crt)
    } else {
        (0, format!("diff: {}", diff), crt)
    }
}

pub fn color(val: f64) -> String {
    let saturation = 0.8;
    let brightness = 0.9;
    let i = (val * 6.0).floor();
    let f = val * 6.0 - i;
    let p = brightness * (1.0 - saturation);
    let q = brightness * (1.0 - f * saturation);
    let t = brightness * (1.0 - (1.0 - f) * saturation);

    let (r, g, b) = match i as i32 % 6 {
        0 => (brightness, t, p),
        1 => (q, brightness, p),
        2 => (p, brightness, t),
        3 => (p, q, brightness),
        4 => (t, p, brightness),
        _ => (brightness, p, q),
    };

    format!("#{:02x}{:02x}{:02x}", (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// 0 <= val <= 1
// pub fn color(mut val: f64) -> String {
//     val.setmin(1.0);
//     val.setmax(0.0);
//     let (r, g, b) = if val < 0.5 {
//         let x = val * 2.0;
//         (
//             30. * (1.0 - x) + 144. * x,
//             144. * (1.0 - x) + 255. * x,
//             255. * (1.0 - x) + 30. * x,
//         )
//     } else {
//         let x = val * 2.0 - 1.0;
//         (
//             144. * (1.0 - x) + 255. * x,
//             255. * (1.0 - x) + 30. * x,
//             30. * (1.0 - x) + 70. * x,
//         )
//     };
//     format!("#{:02x}{:02x}{:02x}", r.round() as i32, g.round() as i32, b.round() as i32)
// }

pub fn rect(x: usize, y: usize, w: usize, h: usize, fill: &str) -> Rectangle {
    Rectangle::new()
        .set("x", x)
        .set("y", y)
        .set("width", w)
        .set("height", h)
        .set("fill", fill)
}

pub fn group(title: String) -> Group {
    Group::new().add(Title::new().add(Text::new(title)))
}

pub fn vis_default(input: &Input, out: &Output) -> (i64, String, String) {
    let (mut score, err, svg) = vis(input, &out.out, out.out.len(), false);
    if err.len() > 0 {
        score = 0;
    }
    (score, err, svg)
}

pub fn vis(input: &Input, out_all: &[usize], t: usize, show_number: bool) -> (i64, String, String) {
    let out = &out_all[..t];
    let (score, err, crt) = compute_score_details(input, &out);
    if input.puzzle_type.starts_with("cube") {
        let n = f64::sqrt((input.n / 6) as f64).round() as usize;
        let D = 800 / (4 * n);
        let W = 4 * n * D * 2 + 50;
        let H = 3 * n * D;
        let mut doc = svg::Document::new()
            .set("id", "vis")
            .set("viewBox", (-5, -5, W + 10, H + 10))
            .set("width", W + 10)
            .set("height", H + 10)
            .set("style", "background-color:white");
        doc = doc.add(Style::new(format!(
            "text {{text-anchor: middle;dominant-baseline: central;}}"
        )));
        let max_c = *input.target.iter().max().unwrap() + 1;
        for (x1, crt) in [
            (0, crt.iter().map(|&p| input.start[p]).collect_vec()),
            (4 * n * D + 50, input.target.clone()),
        ] {
            for (x0, y0, i0) in [
                (n, 0, 0),
                (n, n, n * n),
                (2 * n, n, 2 * n * n),
                (3 * n, n, 3 * n * n),
                (0, n, 4 * n * n),
                (n, 2 * n, 5 * n * n),
            ] {
                for i in 0..n {
                    for j in 0..n {
                        let mut g = group(format!("c[{}] = {}", i0 + i * n + j, crt[i0 + i * n + j])).add(rect(
                            x1 + x0 * D + j * D,
                            y0 * D + i * D,
                            D,
                            D,
                            &color(crt[i0 + i * n + j] as f64 / max_c as f64),
                        ));
                        if show_number {
                            g = g.add(
                                svg::node::element::Text::new()
                                    .add(Text::new(crt[i0 + i * n + j].to_string()))
                                    .set("x", x1 + x0 * D + j * D + D / 2)
                                    .set("y", y0 * D + i * D + D / 2)
                                    .set("font-size", D / 2),
                            );
                        }
                        doc = doc.add(g);
                    }
                }
            }
        }
        (score, err, doc.to_string())
    } else if input.puzzle_type.starts_with("wreath") {
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
        let D = 800 / input.n;
        let W = D * input.n * 2 + 50;
        let H = D * input.n;
        let mut doc = svg::Document::new()
            .set("id", "vis")
            .set("viewBox", (-5, -5, W + 10, H + 10))
            .set("width", W + 10)
            .set("height", H + 10)
            .set("style", "background-color:white");
        doc = doc.add(Style::new(format!(
            "text {{text-anchor: middle;dominant-baseline: central;}}"
        )));
        let max_c = *input.target.iter().max().unwrap() + 1;
        let mut count = vec![0; input.n];
        for i in 0..2 {
            for &j in &loops[i] {
                count[j] += 1;
            }
        }
        for (x1, crt) in [
            (0, crt.iter().map(|&p| input.start[p]).collect_vec()),
            (D * input.n + 50, input.target.clone()),
        ] {
            let size = if input.n <= 10 {
                D * 5 / 10
            } else if input.n <= 12 {
                D * 6 / 10
            } else {
                D * 8 / 10
            };
            let commons = (0..input.n).filter(|&i| count[i] == 2).collect_vec();
            for k in 0..2 {
                for i in 0..loops[k].len() {
                    let x = (x1 + k * D * input.n / 2 + D * input.n / 4) as f64
                        + ((D * input.n / 4 - D) as f64 * f64::cos(2.0 * PI * i as f64 / loops[k].len() as f64)).round();
                    let y = (D * input.n / 4) as f64
                        + ((D * input.n / 4 - D) as f64 * f64::sin(2.0 * PI * i as f64 / loops[k].len() as f64)).round();
                    let mut g = group(format!("c[{}] = {}", loops[k][i], crt[loops[k][i]])).add(
                        Circle::new()
                            .set("cx", x)
                            .set("cy", y)
                            .set("r", size)
                            .set("fill", color(crt[loops[k][i]] as f64 / max_c as f64)),
                    );
                    if show_number {
                        g = g.add(
                            svg::node::element::Text::new()
                                .add(Text::new(crt[loops[k][i]].to_string()))
                                .set("x", x)
                                .set("y", y)
                                .set("font-size", size),
                        );
                    }
                    doc = doc.add(g);
                    if count[loops[k][i]] == 2 {
                        doc = doc.add(
                            Circle::new()
                                .set("cx", x)
                                .set("cy", y)
                                .set(
                                    "r",
                                    if input.n <= 10 {
                                        D * 6 / 10
                                    } else if input.n <= 12 {
                                        D * 7 / 10
                                    } else {
                                        D * 9 / 10
                                    },
                                )
                                .set("fill", "none")
                                .set("stroke", if loops[k][i] == commons[0] { "blue" } else { "green" })
                                .set("stroke-width", 2),
                        );
                    }
                }
            }
        }
        (score, err, doc.to_string())
    } else if input.puzzle_type.starts_with("globe") {
        let rows = input.move_names.iter().filter(|s| s.starts_with("r")).count() as usize;
        let cols = input.move_names.iter().filter(|s| s.starts_with("f")).count() as usize;
        let D = 800 / cols.max(rows);
        let W = D * cols * 2 + 50;
        let H = D * rows;
        let mut doc = svg::Document::new()
            .set("id", "vis")
            .set("viewBox", (-5, -5, W + 10, H + 10 + 30))
            .set("width", W + 10)
            .set("height", H + 10 + 30)
            .set("style", "background-color:white");
        doc = doc.add(Style::new(format!(
            "text {{text-anchor: middle;dominant-baseline: central;}}"
        )));
        let max_c = *input.target.iter().max().unwrap() + 1;
        for (x1, crt) in [
            (0, crt.iter().map(|&p| input.start[p]).collect_vec()),
            (D * cols + 50, input.target.clone()),
        ] {
            for i in 0..rows {
                for j in 0..cols {
                    let mut g = group(format!("c[{}] = {}", i * cols + j, crt[i * cols + j])).add(rect(
                        x1 + j * D,
                        i * D + 30,
                        D,
                        D,
                        &color(crt[i * cols + j] as f64 / max_c as f64),
                    ));
                    if show_number {
                        g = g.add(
                            svg::node::element::Text::new()
                                .add(Text::new(crt[i * cols + j].to_string()))
                                .set("x", x1 + j * D + D / 2)
                                .set("y", i * D + D / 2 + 30)
                                .set("font-size", D / 2),
                        );
                    }
                    doc = doc.add(g);
                }
            }
        }
        let mut cmd = String::new();
        for i in t - t.min(21)..out_all.len().min(t + 20) {
            if cmd.len() > 0 {
                cmd += ".";
            }
            if i == t - 1 {
                cmd += &format!("<tspan fill=\"red\">{}</tspan>", input.move_names[out_all[i]]);
            } else {
                cmd += &input.move_names[out_all[i]];
            }
        }
        doc = doc.add(
            svg::node::element::Text::new()
                .add(Text::new(cmd))
                .set("x", W / 2)
                .set("y", 10)
                .set("font-size", 18)
                .set("fill", "black"),
        );
        (score, err, doc.to_string())
    } else {
        panic!("unknown type: {}", input.puzzle_type);
    }
}
