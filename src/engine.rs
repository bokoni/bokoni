//! Embedding API for Rust.
//!
//! ```
//! use bokoni::Engine;
//!
//! let mut engine = Engine::new();
//!
//! // Rust functions: arguments and return value are converted automatically.
//! engine.register_fn("greet", |name: String| format!("Hello {name}"));
//! engine.register_fn("div", |a: i64, b: i64| {
//!     if b == 0 { Err("division by zero") } else { Ok(a / b) }
//! });
//!
//! let s: String = engine.eval_as(r#"greet("Vincent")"#).unwrap();
//! assert_eq!(s, "Hello Vincent");
//!
//! // Definitions remain available from one call to the next.
//! engine.eval("def square(x: int): int = x * x").unwrap();
//! let v: i64 = engine.call_as("square", (7,)).unwrap();
//! assert_eq!(v, 49);
//! ```

use crate::ast::Program;
use crate::diagnostic::{Error, Errors};
use crate::intern::Interner;
use crate::ir::{Proto, Script};
use crate::value::{Function, NativeFn, NativeImpl, Value};
use crate::vm::{CallArgs, Runtime};
use crate::{parser, printer};
use std::fmt::Display;
use std::rc::Rc;

pub struct Engine {
    rt: Runtime,
    static_checks: bool,
    /// Static-check warnings waiting to be read.
    warnings: Vec<Error>,
    /// Warning handler (otherwise warnings are kept for `take_warnings`).
    on_warning: Option<WarningHandler>,
    /// Provides the files of `require` (see [`Engine::set_loader`]).
    loader: Option<Loader>,
    /// Macros declared by the scripts so far.
    macros: crate::macros::Macros,
    /// Files already loaded (by their name), and those being loaded (cycles).
    loaded: Vec<String>,
    loading: Vec<String>,
}

type WarningHandler = Box<dyn FnMut(&Error)>;
type Loader = Box<dyn FnMut(&str, &str) -> Result<Source, String>>;

/// A file provided to `require` by the loader.
pub struct Source {
    /// Its name: identifies it (a file is loaded once per name) and appears in error
    /// messages; the loader receives it back as the origin of nested `require`s.
    pub name: String,
    pub text: String,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Global functions capture the global scope that contains them:
        // break that cycle to free the memory; then the cycles of values.
        self.rt.globals.clear();
        crate::gc::collect();
    }
}

/// Built-in declarations: the `Deprecated` and `Doc` annotations.
const PRELUDE: &str = "annotation Deprecated\n  message: str? = nil\nend\n\
                       annotation Doc\n  text: str\nend\n\
                       class Error\n  def init(@message: str)\n  end\n  def message = @message\nend\n";

impl Engine {
    pub fn new() -> Self {
        let mut e = Engine {
            rt: Runtime::new(),
            static_checks: true,
            warnings: Vec::new(),
            on_warning: None,
            loader: None,
            loaded: Vec::new(),
            loading: Vec::new(),
            macros: Default::default(),
        };
        e.eval_named(PRELUDE, "<prelude>")
            .expect("the prelude compiles");
        e
    }

    /// Provides the files of `require "name"`: `loader(name, from)` receives the name
    /// as written and the name of the script that requires it (to resolve relative
    /// paths), and returns the file. Without a loader, `require` is an error.
    ///
    /// ```
    /// use bokoni::{Engine, Source};
    ///
    /// let mut engine = Engine::new();
    /// engine.set_loader(|name, _from| match name {
    ///     "math" => Ok(Source {
    ///         name: "math.bk".into(),
    ///         text: "module Math\n  def sq(x: int): int = x * x\nend".into(),
    ///     }),
    ///     _ => Err("no such file".into()),
    /// });
    /// let v: i64 = engine.eval_as("require \"math\"\nMath.sq(7)").unwrap();
    /// assert_eq!(v, 49);
    /// ```
    pub fn set_loader(&mut self, f: impl FnMut(&str, &str) -> Result<Source, String> + 'static) {
        self.loader = Some(Box::new(f));
    }

    /// Loads the files required by a script (each one once, in order), before it is
    /// checked: their declarations are then known, as those of a previous script.
    fn require(&mut self, program: &crate::ast::Program, from: &str) -> Result<(), Errors> {
        for s in &program.body.stmts {
            let crate::ast::StmtKind::Require(path) = s.kind else {
                continue;
            };
            let path = self.rt.interner.resolve(path).to_string();
            let fail = |msg: String| -> Errors { Error::runtime(msg, s.span).into() };
            let Some(mut loader) = self.loader.take() else {
                return Err(fail(format!(
                    "`require \"{path}\"`: no file loader (`Engine::set_loader`)"
                )));
            };
            let found = loader(&path, from);
            self.loader = Some(loader);
            let src = found.map_err(|e| fail(format!("cannot load `{path}`: {e}")))?;
            if self.loaded.contains(&src.name) {
                continue;
            }
            if self.loading.contains(&src.name) {
                let mut chain = self.loading.clone();
                chain.push(src.name);
                return Err(fail(format!("circular `require`: {}", chain.join(" → "))));
            }
            self.loading.push(src.name.clone());
            let r = self.eval_named(&src.text, &src.name);
            self.loading.pop();
            r?;
            self.loaded.push(src.name);
        }
        Ok(())
    }

    /// Redirects the output of `print` (default: standard output).
    pub fn set_output(&mut self, f: impl FnMut(&str) + 'static) {
        self.rt.output = Box::new(f);
    }

    /// Static-check warnings since the last call (use of a `@[Deprecated]`
    /// declaration…): they do not prevent execution.
    pub fn take_warnings(&mut self) -> Vec<Error> {
        std::mem::take(&mut self.warnings)
    }

    /// Receives each warning as soon as checking produces it, before the script runs
    /// (instead of keeping them for [`take_warnings`](Self::take_warnings)).
    pub fn set_warning_handler(&mut self, f: impl FnMut(&Error) + 'static) {
        self.on_warning = Some(Box::new(f));
    }

    /// Enables or disables static checking of names and types before execution
    /// (enabled by default). Runtime checks stay active.
    pub fn set_static_checks(&mut self, enabled: bool) {
        self.static_checks = enabled;
    }

    /// Enables or disables native (JIT) compilation of typed functions (enabled by
    /// default when the `jit` feature is compiled in). The result is the same; only
    /// the speed changes.
    pub fn set_jit(&mut self, enabled: bool) {
        self.rt.jit_enabled = enabled && cfg!(feature = "jit");
    }

    /// Compiles each function on its first call (instead of the second): used by the
    /// equivalence tests between native code and the interpreter.
    #[doc(hidden)]
    pub fn set_jit_eager(&mut self, eager: bool) {
        self.rt.jit_eager = eager;
    }

    /// Whether the global function `name` was compiled to native code (after at least one
    /// call). For a `forall` function: one of its copies typed for the arguments of a call
    /// (`T = int`).
    pub fn is_native(&self, name: &str) -> bool {
        let native =
            |p: &crate::ir::Proto| matches!(*p.jit.borrow(), crate::ir::JitState::Native(_));
        match self.get_global(name).as_ref() {
            Some(Value::Function(f)) => match &**f {
                Function::Script(c) => {
                    native(&c.proto) || c.proto.specs.borrow().iter().any(|(_, s)| native(s))
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether method `method` of type `ty` was compiled to native code (after at least one
    /// call).
    pub fn is_native_method(&self, ty: &str, method: &str) -> bool {
        let (Some(Value::Type(info)), Some(m)) =
            (&self.get_global(ty), self.rt.interner.get(method))
        else {
            return false;
        };
        let native =
            |p: &crate::ir::Proto| matches!(*p.jit.borrow(), crate::ir::JitState::Native(_));
        info.methods
            .get(&m)
            .is_some_and(|p| native(p) || p.specs.borrow().iter().any(|(_, s)| native(s)))
    }

    /// Maximum call depth before the "infinite recursion?" error (default: 1000).
    pub fn set_max_call_depth(&mut self, depth: usize) {
        self.rt.max_depth = depth;
    }

    /// Registers a Rust function. Its arguments (up to 6) and its return value are
    /// converted automatically; it may return `Result<T, E: Display>`.
    pub fn register_fn<Args>(&mut self, name: &str, f: impl IntoNative<Args>) {
        let name: Rc<str> = name.into();
        let native = NativeFn {
            f: f.into_native(name.clone()),
            name: name.clone(),
        };
        self.define(&name, Value::Function(Rc::new(Function::Native(native))));
    }

    /// Registers a Rust function that receives the raw values.
    pub fn register_raw(
        &mut self,
        name: &str,
        f: impl Fn(&[Value]) -> Result<Value, String> + 'static,
    ) {
        let native = NativeFn {
            name: name.into(),
            f: Box::new(f),
        };
        self.define(name, Value::Function(Rc::new(Function::Native(native))));
    }

    fn define(&mut self, name: &str, v: Value) {
        let sym = self.rt.interner.intern(name);
        self.rt.globals.define(sym, v, false);
    }

    /// Defines (or replaces) a global variable the script can modify.
    pub fn set_global(&mut self, name: &str, v: impl IntoValue) {
        let sym = self.rt.interner.intern(name);
        let v = v.into_value();
        crate::gc::track_new(&v);
        self.rt.globals.define(sym, v, true);
    }

    /// Reads a global variable.
    pub fn get_global(&self, name: &str) -> Option<Value> {
        let sym = self.rt.interner.get(name)?;
        self.rt.globals.get(sym)
    }

    // ─── Introspection ───────────────────────────────────────────────────────

    fn not_found(name: &str, what: &str) -> Errors {
        Error::new(
            crate::ErrorKind::Runtime,
            format!("`{name}` is not {what}"),
            None,
        )
        .into()
    }

    /// Description of a `type`, `class` or `annotation`: fields, methods,
    /// annotations (their arguments are evaluated now).
    pub fn type_info(&mut self, name: &str) -> Result<crate::reflect::TypeDesc, Errors> {
        match self.get_global(name).as_ref() {
            Some(Value::Type(info)) => Ok(self.rt.type_desc(info)?),
            _ => Err(Self::not_found(name, "a declared type")),
        }
    }

    /// Description of a script function: parameters, return type, annotations.
    pub fn function_info(&mut self, name: &str) -> Result<crate::reflect::FnDesc, Errors> {
        match self.get_global(name).as_ref() {
            Some(Value::Function(f)) => match &**f {
                crate::value::Function::Script(c) => {
                    let p = c.proto.clone();
                    Ok(self.rt.fn_desc(name, &p)?)
                }
                _ => Err(Self::not_found(name, "a script function")),
            },
            _ => Err(Self::not_found(name, "a script function")),
        }
    }

    /// Description of an `enum`: variants, methods, annotations.
    pub fn enum_info(&mut self, name: &str) -> Result<crate::reflect::EnumDesc, Errors> {
        match self.get_global(name).as_ref() {
            Some(Value::Enum(info)) => Ok(self.rt.enum_desc(info)?),
            _ => Err(Self::not_found(name, "a declared enum")),
        }
    }

    /// Runs code. Its `let`s and `def`s stay visible to later calls.
    ///
    /// Syntax errors are all reported at once (`Errors`); a runtime error stops the
    /// script and is reported alone.
    pub fn eval(&mut self, src: &str) -> Result<Value, Errors> {
        self.eval_named(src, "<eval>")
    }

    /// Like [`eval`](Self::eval), naming the script for error messages.
    pub fn eval_named(&mut self, src: &str, name: &str) -> Result<Value, Errors> {
        let main = self.compile(src, name)?;
        let r = self.rt.run(&main);
        crate::gc::maybe_collect();
        Ok(r?)
    }

    /// Frees the values only reachable from reference cycles (a closure that captures
    /// itself, objects that point to each other…), now. Also done automatically, after a
    /// script and during one, when many values were created. Returns how many containers
    /// were freed.
    pub fn collect_cycles(&mut self) -> usize {
        crate::gc::collect()
    }

    /// Parses a script: the program and the script (its source followed by the code its
    /// macro calls generated).
    fn parse(&mut self, src: &str, name: &str) -> Result<(Program, Rc<Script>), Errors> {
        let r = parser::parse_with(src, &mut self.rt.interner, &mut self.macros);
        let script = Rc::new(Script {
            src: self.macros.full.as_str().into(),
            name: name.into(),
        });
        match r {
            Ok(p) => Ok((p, script)),
            Err(mut errs) => {
                for e in &mut errs.0 {
                    e.src = Some((script.name.clone(), script.src.clone()));
                }
                Err(errs)
            }
        }
    }

    /// Parses, checks and compiles a script to IR.
    fn compile(&mut self, src: &str, name: &str) -> Result<Rc<Proto>, Errors> {
        let (mut program, mut script) = self.parse(src, name)?;
        // The script itself counts as being loaded (a file that requires it: a cycle).
        let top = !self.loading.iter().any(|n| n == name);
        if top {
            self.loading.push(name.into());
        }
        let macros = self.macros.defs.len();
        let r = self.require(&program, name);
        if top {
            self.loading.pop();
        }
        r.map_err(|mut errs| {
            // Errors of the `require` itself are in this script.
            for e in &mut errs.0 {
                if e.src.is_none() {
                    e.src = Some((script.name.clone(), script.src.clone()));
                }
            }
            errs
        })?;
        // The required files declared macros: parsed again, so that it can use them.
        if self.macros.defs.len() != macros {
            (program, script) = self.parse(src, name)?;
        }
        let with_src = |mut errs: Errors| {
            for e in &mut errs.0 {
                e.src = Some((script.name.clone(), script.src.clone()));
            }
            errs
        };
        crate::compile::intern_module_names(&program, &mut self.rt.interner, &mut self.rt.globals);
        let prims = if self.static_checks {
            let (errors, warnings, prims) =
                crate::check::check(&program, &self.rt.interner, &self.rt.globals);
            for mut w in warnings {
                w.src = Some((script.name.clone(), script.src.clone()));
                match &mut self.on_warning {
                    Some(f) => f(&w),
                    None => self.warnings.push(w),
                }
            }
            if !errors.is_empty() {
                return Err(with_src(Errors::new(errors)));
            }
            Some(prims)
        } else {
            None
        };
        let mut main = crate::compile::compile(
            &program,
            &script,
            &mut self.rt.interner,
            &mut self.rt.globals,
            prims.as_ref(),
        )
        .map_err(with_src)?;
        // Loops of the main program as functions, for the typed JIT.
        if cfg!(feature = "jit")
            && self.rt.jit_enabled
            && let Some(p) = Rc::get_mut(&mut main)
        {
            let name = self.rt.interner.intern("_");
            crate::outline::outline_loops(p, name);
        }
        Ok(main)
    }

    /// IR of the code (the representation the interpreter runs), as readable text.
    pub fn ir(&mut self, src: &str) -> Result<String, Errors> {
        let main = self.compile(src, "<ir>")?;
        Ok(main.dump(&self.rt.interner))
    }

    /// Runs code and converts the result.
    pub fn eval_as<T: FromValue>(&mut self, src: &str) -> Result<T, Errors> {
        let v = self.eval(src)?;
        T::from_value(&v).map_err(|msg| Error::new(crate::ErrorKind::Type, msg, None).into())
    }

    /// Calls a script (or registered) function by name.
    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, Errors> {
        let f = self.get_global(name).ok_or_else(|| {
            Error::new(
                crate::ErrorKind::Runtime,
                format!("`{name}` is not defined"),
                None,
            )
        })?;
        args.iter().for_each(crate::gc::track_new);
        Ok(self
            .rt
            .call_value(&f, CallArgs::plain(args), Default::default())?)
    }

    /// Calls a function with Rust arguments (a tuple) and converts the result.
    pub fn call_as<T: FromValue>(&mut self, name: &str, args: impl IntoArgs) -> Result<T, Errors> {
        let v = self.call(name, args.into_args())?;
        T::from_value(&v).map_err(|msg| Error::new(crate::ErrorKind::Type, msg, None).into())
    }

    /// AST of the code, as S-expressions (to debug the parser).
    pub fn ast(&mut self, src: &str) -> Result<String, Errors> {
        let program = parser::parse(src, &mut self.rt.interner)?;
        Ok(printer::program(&program, &self.rt.interner))
    }

    /// Source-code representation of a value (`"a"`, `[1, 2]`, `{x: 1}`).
    pub fn inspect(&self, v: &Value) -> String {
        v.inspect(&self.rt.interner)
    }

    /// Text of a value as `print` would show it.
    pub fn display(&self, v: &Value) -> String {
        v.to_display(&self.rt.interner)
    }

    pub fn interner(&self) -> &Interner {
        &self.rt.interner
    }
}

// ─── Rust ⇄ script conversions ──────────────────────────────────────────────

/// Conversion from a script value to a Rust type.
pub trait FromValue: Sized {
    fn from_value(v: &Value) -> Result<Self, String>;
}

/// Conversion from a Rust type to a script value.
pub trait IntoValue {
    fn into_value(self) -> Value;
}

fn mismatch(expected: &str, v: &Value) -> String {
    format!("expected {expected}, got {}", v.type_name())
}

impl FromValue for Value {
    fn from_value(v: &Value) -> Result<Self, String> {
        Ok(v.clone())
    }
}

impl IntoValue for Value {
    fn into_value(self) -> Value {
        self
    }
}

macro_rules! int_conv {
    ($($t:ty),*) => {$(
        impl FromValue for $t {
            fn from_value(v: &Value) -> Result<Self, String> {
                match v {
                    Value::Int(i) => <$t>::try_from(*i)
                        .map_err(|_| format!("{i} does not fit in {}", stringify!($t))),
                    _ => Err(mismatch("int", v)),
                }
            }
        }
        impl IntoValue for $t {
            fn into_value(self) -> Value {
                // Integers larger than i64 are saturated.
                Value::Int(i64::try_from(self).unwrap_or(i64::MAX))
            }
        }
    )*};
}
int_conv!(i8, i16, i32, i64, u8, u16, u32, u64, usize, isize);

impl FromValue for f64 {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(mismatch("float", v)),
        }
    }
}

impl FromValue for f32 {
    fn from_value(v: &Value) -> Result<Self, String> {
        f64::from_value(v).map(|f| f as f32)
    }
}

impl IntoValue for f64 {
    fn into_value(self) -> Value {
        Value::Float(self)
    }
}

impl IntoValue for f32 {
    fn into_value(self) -> Value {
        Value::Float(self as f64)
    }
}

impl FromValue for bool {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v {
            Value::Bool(b) => Ok(*b),
            _ => Err(mismatch("bool", v)),
        }
    }
}

impl IntoValue for bool {
    fn into_value(self) -> Value {
        Value::Bool(self)
    }
}

impl FromValue for String {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v {
            Value::Str(s) => Ok(s.to_string()),
            _ => Err(mismatch("str", v)),
        }
    }
}

impl FromValue for Rc<str> {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v {
            Value::Str(s) => Ok(s.clone()),
            _ => Err(mismatch("str", v)),
        }
    }
}

impl IntoValue for String {
    fn into_value(self) -> Value {
        Value::Str(self.into())
    }
}

impl IntoValue for &str {
    fn into_value(self) -> Value {
        Value::Str(self.into())
    }
}

impl IntoValue for Rc<str> {
    fn into_value(self) -> Value {
        Value::Str(self)
    }
}

impl FromValue for () {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v {
            Value::Nil => Ok(()),
            _ => Err(mismatch("nil", v)),
        }
    }
}

impl IntoValue for () {
    fn into_value(self) -> Value {
        Value::Nil
    }
}

impl<T: FromValue> FromValue for Option<T> {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v {
            Value::Nil => Ok(None),
            v => T::from_value(v).map(Some),
        }
    }
}

impl<T: IntoValue> IntoValue for Option<T> {
    fn into_value(self) -> Value {
        self.map_or(Value::Nil, T::into_value)
    }
}

impl<T: FromValue> FromValue for Vec<T> {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v {
            Value::List(l) => l
                .borrow()
                .iter()
                .enumerate()
                .map(|(i, x)| T::from_value(x).map_err(|e| format!("element {i}: {e}")))
                .collect(),
            _ => Err(mismatch("list", v)),
        }
    }
}

impl<T: IntoValue> IntoValue for Vec<T> {
    fn into_value(self) -> Value {
        Value::list(self.into_iter().map(T::into_value).collect())
    }
}

/// Return value of a Rust function: `T` or `Result<T, E>`.
pub trait NativeReturn {
    fn into_result(self) -> Result<Value, String>;
}

impl<T: IntoValue> NativeReturn for T {
    fn into_result(self) -> Result<Value, String> {
        Ok(self.into_value())
    }
}

impl<T: IntoValue, E: Display> NativeReturn for Result<T, E> {
    fn into_result(self) -> Result<Value, String> {
        self.map(T::into_value).map_err(|e| e.to_string())
    }
}

/// Rust function convertible into a script function. `Args` is the argument tuple.
pub trait IntoNative<Args> {
    fn into_native(self, name: Rc<str>) -> Box<NativeImpl>;
}

/// Rust arguments passed to [`Engine::call_as`]: a tuple of convertible values.
pub trait IntoArgs {
    fn into_args(self) -> Vec<Value>;
}

macro_rules! native {
    ($n:literal $(, $t:ident $v:ident)*) => {
        impl<F, R $(, $t)*> IntoNative<($($t,)*)> for F
        where
            F: Fn($($t),*) -> R + 'static,
            R: NativeReturn,
            $($t: FromValue,)*
        {
            fn into_native(self, name: Rc<str>) -> Box<NativeImpl> {
                Box::new(move |args: &[Value]| {
                    if args.len() != $n {
                        return Err(format!(
                            "`{name}` expects {} argument{}, got {}",
                            $n,
                            if $n == 1 { "" } else { "s" },
                            args.len()
                        ));
                    }
                    #[allow(unused_mut, unused_variables)]
                    let mut it = args.iter().enumerate();
                    $(
                        let (i, a) = it.next().unwrap();
                        let $v = $t::from_value(a)
                            .map_err(|e| format!("argument {} of `{name}`: {e}", i + 1))?;
                    )*
                    (self)($($v),*).into_result()
                })
            }
        }

        impl<$($t: IntoValue),*> IntoArgs for ($($t,)*) {
            #[allow(clippy::vec_init_then_push)]
            fn into_args(self) -> Vec<Value> {
                #[allow(non_snake_case)]
                let ($($v,)*) = self;
                vec![$($v.into_value()),*]
            }
        }
    };
}

native!(0);
native!(1, A a);
native!(2, A a, B b);
native!(3, A a, B b, C c);
native!(4, A a, B b, C c, D d);
native!(5, A a, B b, C c, D d, E e);
native!(6, A a, B b, C c, D d, E e, G g);
