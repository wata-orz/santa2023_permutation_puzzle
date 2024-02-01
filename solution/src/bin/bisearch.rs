#![allow(non_snake_case)]

use solution::*;

fn main() {
    get_time();
    let input = read_input();
    if !input.puzzle_type.starts_with("globe") {
        return;
    }
    let out = read_best(&input);
    let out = optimize_bisearch(&input, &out);
    let out = optimize_bisearch_globe(&input, &out);
    write_output(&input, &out);
}
