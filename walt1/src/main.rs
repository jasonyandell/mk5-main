// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

use std::io::Read;

fn main() {
    let mut req = String::new();
    if std::io::stdin().read_to_string(&mut req).is_err() {
        println!("{{\"v\":1,\"error\":\"the request is not UTF-8 text\"}}");
        return;
    }
    println!("{}", walt::handle(&req));
}
