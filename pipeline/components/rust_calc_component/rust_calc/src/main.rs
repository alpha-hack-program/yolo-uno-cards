use clap::Parser;
use rust_decimal::Decimal;

/// Simple program to multiply two floating-point numbers
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// First floating-point number
    #[arg(long)]
    num1: f64,

    /// Second floating-point number
    #[arg(long)]
    num2: f64,

    /// Output file to write the result
    #[arg(short, long)]
    output: String,
}

fn main() {
    let args = Args::parse();

    // Multiply the two numbers

    let num1 = Decimal::from_f64_retain(args.num1).unwrap();
    let num2 = Decimal::from_f64_retain(args.num2).unwrap();

    let result = num1 * num2;

    println!("Result: {} * {} = {:?}", num1, num2, result);

    // Write the result to the output file
    std::fs::write(&args.output, result.to_string())
        .expect("Failed to write the output file");

    println!("Successfully wrote the result to {}", args.output);
}
