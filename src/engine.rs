//! API d'intégration dans Rust.
//!
//! ```
//! use bokoni::Engine;
//!
//! let mut engine = Engine::new();
//!
//! // Fonctions Rust : arguments et retour convertis automatiquement.
//! engine.register_fn("greet", |name: String| format!("Bonjour {name}"));
//! engine.register_fn("div", |a: i64, b: i64| {
//!     if b == 0 { Err("division par zéro") } else { Ok(a / b) }
//! });
//!
//! let s: String = engine.eval_as(r#"greet("Vincent")"#).unwrap();
//! assert_eq!(s, "Bonjour Vincent");
//!
//! // Les définitions restent disponibles d'un appel à l'autre.
//! engine.eval("def square(x: int): int = x * x").unwrap();
//! let v: i64 = engine.call_as("square", (7,)).unwrap();
//! assert_eq!(v, 49);
//! ```

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
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Les fonctions globales capturent la portée globale qui les contient :
        // on casse ce cycle pour libérer la mémoire.
        self.rt.globals.clear();
    }
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            rt: Runtime::new(),
            static_checks: true,
        }
    }

    /// Redirige la sortie de `print` (par défaut : la sortie standard).
    pub fn set_output(&mut self, f: impl FnMut(&str) + 'static) {
        self.rt.output = Box::new(f);
    }

    /// Active ou désactive la vérification statique des noms et des types avant
    /// l'exécution (activée par défaut). Les vérifications à l'exécution restent actives.
    pub fn set_static_checks(&mut self, enabled: bool) {
        self.static_checks = enabled;
    }

    /// Active ou désactive la compilation native (JIT) des fonctions typées (activée par
    /// défaut quand la fonctionnalité `jit` est compilée). Le résultat est le même ; seule
    /// la vitesse change.
    pub fn set_jit(&mut self, enabled: bool) {
        self.rt.jit_enabled = enabled && cfg!(feature = "jit");
    }

    /// La fonction globale `name` a-t-elle été compilée en code natif (après au moins un
    /// appel) ?
    pub fn is_native(&self, name: &str) -> bool {
        match self.get_global(name) {
            Some(Value::Function(f)) => match &*f {
                Function::Script(c) => {
                    matches!(*c.proto.jit.borrow(), crate::ir::JitState::Native(_))
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Profondeur d'appels maximale avant l'erreur « récursion infinie ? » (défaut : 1000).
    pub fn set_max_call_depth(&mut self, depth: usize) {
        self.rt.max_depth = depth;
    }

    /// Enregistre une fonction Rust. Ses arguments (jusqu'à 6) et sa valeur de retour
    /// sont convertis automatiquement ; elle peut renvoyer `Result<T, E: Display>`.
    pub fn register_fn<Args>(&mut self, name: &str, f: impl IntoNative<Args>) {
        let name: Rc<str> = name.into();
        let native = NativeFn {
            f: f.into_native(name.clone()),
            name: name.clone(),
        };
        self.define(&name, Value::Function(Rc::new(Function::Native(native))));
    }

    /// Enregistre une fonction Rust qui reçoit les valeurs brutes.
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

    /// Définit (ou remplace) une variable globale modifiable par le script.
    pub fn set_global(&mut self, name: &str, v: impl IntoValue) {
        let sym = self.rt.interner.intern(name);
        self.rt.globals.define(sym, v.into_value(), true);
    }

    /// Lit une variable globale.
    pub fn get_global(&self, name: &str) -> Option<Value> {
        let sym = self.rt.interner.get(name)?;
        self.rt.globals.get(sym)
    }

    /// Exécute du code. Ses `let` et `def` restent visibles pour les appels suivants.
    ///
    /// Les erreurs de syntaxe sont toutes signalées en une fois (`Errors`) ; une erreur
    /// d'exécution arrête le script et est seule.
    pub fn eval(&mut self, src: &str) -> Result<Value, Errors> {
        self.eval_named(src, "<eval>")
    }

    /// Comme [`eval`](Self::eval), en nommant le script pour les messages d'erreur.
    pub fn eval_named(&mut self, src: &str, name: &str) -> Result<Value, Errors> {
        let main = self.compile(src, name)?;
        Ok(self.rt.run(&main)?)
    }

    /// Analyse, vérifie et compile un script en IR.
    fn compile(&mut self, src: &str, name: &str) -> Result<Rc<Proto>, Errors> {
        let script = Rc::new(Script {
            src: src.into(),
            name: name.into(),
        });
        let with_src = |mut errs: Errors| {
            for e in &mut errs.0 {
                e.src = Some((script.name.clone(), script.src.clone()));
            }
            errs
        };
        let program = parser::parse(src, &mut self.rt.interner).map_err(with_src)?;
        let prims = if self.static_checks {
            let (errors, prims) =
                crate::check::check(&program, &self.rt.interner, &self.rt.globals);
            if !errors.is_empty() {
                return Err(with_src(Errors::new(errors)));
            }
            Some(prims)
        } else {
            None
        };
        crate::compile::compile(
            &program,
            &script,
            &mut self.rt.interner,
            &mut self.rt.globals,
            prims.as_ref(),
        )
        .map_err(with_src)
    }

    /// IR du code (représentation exécutée par l'interpréteur), en texte lisible.
    pub fn ir(&mut self, src: &str) -> Result<String, Errors> {
        let main = self.compile(src, "<ir>")?;
        Ok(main.dump(&self.rt.interner))
    }

    /// Exécute du code et convertit le résultat.
    pub fn eval_as<T: FromValue>(&mut self, src: &str) -> Result<T, Errors> {
        let v = self.eval(src)?;
        T::from_value(&v).map_err(|msg| Error::new(crate::ErrorKind::Type, msg, None).into())
    }

    /// Appelle une fonction du script (ou enregistrée) par son nom.
    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, Errors> {
        let f = self.get_global(name).ok_or_else(|| {
            Error::new(
                crate::ErrorKind::Runtime,
                format!("`{name}` n'est pas défini"),
                None,
            )
        })?;
        Ok(self
            .rt
            .call_value(&f, CallArgs::plain(args), Default::default())?)
    }

    /// Appelle une fonction avec des arguments Rust (un tuple) et convertit le résultat.
    pub fn call_as<T: FromValue>(&mut self, name: &str, args: impl IntoArgs) -> Result<T, Errors> {
        let v = self.call(name, args.into_args())?;
        T::from_value(&v).map_err(|msg| Error::new(crate::ErrorKind::Type, msg, None).into())
    }

    /// AST du code, en S-expressions (pour déboguer le parseur).
    pub fn ast(&mut self, src: &str) -> Result<String, Errors> {
        let program = parser::parse(src, &mut self.rt.interner)?;
        Ok(printer::program(&program, &self.rt.interner))
    }

    /// Représentation d'une valeur en code source (`"a"`, `[1, 2]`, `{x: 1}`).
    pub fn inspect(&self, v: &Value) -> String {
        v.inspect(&self.rt.interner)
    }

    /// Texte d'une valeur tel que l'afficherait `print`.
    pub fn display(&self, v: &Value) -> String {
        v.to_display(&self.rt.interner)
    }

    pub fn interner(&self) -> &Interner {
        &self.rt.interner
    }
}

// ─── Conversions Rust ⇄ script ──────────────────────────────────────────────

/// Conversion d'une valeur du script vers un type Rust.
pub trait FromValue: Sized {
    fn from_value(v: &Value) -> Result<Self, String>;
}

/// Conversion d'un type Rust vers une valeur du script.
pub trait IntoValue {
    fn into_value(self) -> Value;
}

fn mismatch(expected: &str, v: &Value) -> String {
    format!("{expected} attendu, reçu {}", v.type_name())
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
                        .map_err(|_| format!("{i} ne tient pas dans un {}", stringify!($t))),
                    _ => Err(mismatch("int", v)),
                }
            }
        }
        impl IntoValue for $t {
            fn into_value(self) -> Value {
                // Les entiers plus grands que i64 sont saturés.
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
                .map(|(i, x)| T::from_value(x).map_err(|e| format!("élément {i} : {e}")))
                .collect(),
            _ => Err(mismatch("liste", v)),
        }
    }
}

impl<T: IntoValue> IntoValue for Vec<T> {
    fn into_value(self) -> Value {
        Value::list(self.into_iter().map(T::into_value).collect())
    }
}

/// Valeur de retour d'une fonction Rust : `T` ou `Result<T, E>`.
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

/// Fonction Rust convertible en fonction du script. `Args` est le tuple des arguments.
pub trait IntoNative<Args> {
    fn into_native(self, name: Rc<str>) -> Box<NativeImpl>;
}

/// Arguments Rust passés à [`Engine::call_as`] : un tuple de valeurs convertibles.
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
                            "`{name}` attend {} argument{}, reçu {}",
                            $n,
                            if $n > 1 { "s" } else { "" },
                            args.len()
                        ));
                    }
                    #[allow(unused_mut, unused_variables)]
                    let mut it = args.iter().enumerate();
                    $(
                        let (i, a) = it.next().unwrap();
                        let $v = $t::from_value(a)
                            .map_err(|e| format!("argument {} de `{name}` : {e}", i + 1))?;
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
