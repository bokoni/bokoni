/// Generates `lines` lines of code. `names` = number of distinct variable names
/// (`usize::MAX`: a different name on every line, the worst case for interning).
fn generate(lines: usize, names: usize) -> String {
    let mut src = String::with_capacity(lines * 64);
    for i in 0..lines {
        let v = i % names;
        match i % 3 {
            0 => src.push_str(&format!(
                "let v{v} = foo(a + b * {i}, [1, 2, x.y], \"text\") - (c - 4) / bar.baz({i})\n"
            )),
            1 => src.push_str(&format!("total = total + compute(i, items[{i}], -k) * 2\n")),
            _ => src.push_str(&format!("ok = check(v{v}.size * 3 + 1 < limit, name) == flag\n")),
        }
    }
    src
}
