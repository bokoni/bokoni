//! Bokoni: a scripting language embedded in Rust, with Ruby-like syntax and gradual typing.
//!
//! ```
//! use bokoni::Engine;
//!
//! let mut engine = Engine::new();
//! engine.register_fn("double", |x: i64| x * 2);
//! let v: i64 = engine.eval_as("def add(a: int, b: int): int = a + b\nadd(1, double(20))").unwrap();
//! assert_eq!(v, 41);
//! ```

pub mod ast;
pub mod check;
pub mod compile;
pub mod diagnostic;
pub mod engine;
pub mod globals;
pub mod intern;
pub mod ir;
#[cfg(feature = "jit")]
pub mod jit;
pub mod lexer;
pub mod objects;
pub mod ops;
pub mod parser;
pub mod printer;
pub mod reflect;
pub mod span;
pub mod value;
pub mod vm;

pub use ast::Visibility;
pub use diagnostic::{Error, ErrorKind, Errors};
pub use engine::{Engine, FromValue, IntoValue, Source};
pub use reflect::{
    Annotation, DeclKind, EnumDesc, FieldDesc, FnDesc, ParamDesc, TypeDesc, VariantDesc,
};
pub use value::Value;
