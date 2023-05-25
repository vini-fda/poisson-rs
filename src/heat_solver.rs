use crate::conj_grad::conjugate_gradient;
use crate::csr::poisson_matrix;
use plotters::prelude::*;

pub fn build_initial_condition<F>(f: F, m: usize, h: f32, x0: f32, y0: f32) -> Vec<f32>
where F: Fn(f32, f32) -> f32 {
    let mut u0 = vec![0.0; m*m];
    for i in 0..m {
        for j in 0..m {
            let x = i as f32 * h - x0;
            let y = j as f32 * h - y0;
            u0[i + j * m] = f(x, y);
        }
    }
    u0
}

pub fn solve_heat(m: usize, alpha: f32, h: f32, dt: f32, n_iter: usize, u0: &[f32]) -> Vec<f32> {
    let a = poisson_matrix(m, dt, h, alpha);
    let mut u = u0.to_vec();
    let mut u_next = u0.to_vec();
    let mut tmp = vec![0.0; m*m];
    for _ in 0..n_iter {
        tmp.copy_from_slice(&u_next);
        conjugate_gradient(&a, &u, &mut u_next);
        u.copy_from_slice(&tmp);
    }
    u_next
}

const OUT_FILE_NAME: &str = "3d-plot-heat.gif";
pub fn plot_solution(m: usize, h: f32, u: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::gif(OUT_FILE_NAME, (600, 400), 100)?.into_drawing_area();
    let l = m as f64;
    for pitch in 0..157 {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("2D Heat Equation Solution", ("sans-serif", 20))
            .build_cartesian_3d(0.0..l, 0.0..6.0, 0.0..l)?;
        chart.with_projection(|mut p| {
            p.pitch = 1.57 - (1.57 - pitch as f64 / 50.0).abs();
            p.scale = 0.7;
            p.into_matrix() // build the projection matrix
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(3)
            .draw()?;

        chart.draw_series(
            SurfaceSeries::xoz(
                (0..m).map(|x| x as f64),
                (0..m).map(|x| x as f64),
                |x, y| { u[x as usize + y as usize * m] as f64},
            )
            .style_func(&|&v| {
                (&HSLColor(240.0 / 360.0 - 240.0 / 360.0 * v / 5.0, 1.0, 0.7)).into()
            }),
        )?;

        root.present()?;
    }

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}