use poisson_rs::heat_solver::{build_initial_condition, solve_heat, plot_solution};

fn pdf(x: f32, y: f32) -> f32 {
    const SDX: f32 = 0.1;
    const SDY: f32 = 0.1;
    const A: f32 = 5.0;
    let x = x / 10.0;
    let y = y / 10.0;
    A * (-x * x / 2.0 / SDX / SDX - y * y / 2.0 / SDY / SDY).exp()
}

fn main() {
    const M: usize = 100;
    const H: f32 = 0.05;
    const ALPHA: f32 = 1.0;
    const DT: f32 = 0.1;
    const N_ITER: usize = 3;
    let x0 = M as f32 * H / 2.0;
    let y0 = M as f32 * H / 2.0;
    let u0 = build_initial_condition(|x, y| { pdf(x,y) }, M, H, x0, y0);
    let u_soln = solve_heat(M, ALPHA, H, DT, N_ITER, &u0);
    let r = plot_solution(M, H, &u_soln);
    match r {
        Ok(_) => println!("Plot saved successfully"),
        Err(e) => println!("Error: {}", e),
    }
}
