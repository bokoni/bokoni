//! Introspection from Rust: the script's declarations and their annotations.
//!
//! ```
//! use bokoni::Engine;
//!
//! let mut engine = Engine::new();
//! engine
//!     .eval(
//!         "annotation Validate\n  min: int? = nil\n  max: int? = nil\nend\n\
//!          type Server\n  @[Validate(min: 1, max: 65535)]\n  port: int = 80\nend",
//!     )
//!     .unwrap();
//! let ty = engine.type_info("Server").unwrap();
//! let port = ty.field("port").unwrap();
//! let v = port.annotation("Validate").unwrap();
//! assert_eq!(v.get::<Option<i64>>("max").unwrap(), Some(65535));
//! ```
//!
//! Descriptions are copies: they keep no link to the engine. Annotation arguments are
//! evaluated when read.

use crate::ast::Visibility;
use crate::engine::FromValue;
use crate::ir::{Annotations, Proto};
use crate::objects::{EnumInfo, TypeInfo};
use crate::ops::{R, type_err};
use crate::value::Value;
use crate::vm::{CallArgs, Runtime};
use std::rc::Rc;

/// An applied annotation (`@[Validate(min: 1)]`): its name and the values of its fields.
#[derive(Clone, Debug)]
pub struct Annotation {
    name: String,
    fields: Vec<(String, Value)>,
}

impl Annotation {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The annotation's fields, in declaration order (default values included).
    pub fn fields(&self) -> &[(String, Value)] {
        &self.fields
    }

    /// Value of a field.
    pub fn value(&self, field: &str) -> Option<&Value> {
        self.fields.iter().find(|(n, _)| n == field).map(|(_, v)| v)
    }

    /// Value of a field, converted (`v.get::<Option<i64>>("min")`).
    pub fn get<T: FromValue>(&self, field: &str) -> Result<T, String> {
        match self.value(field) {
            Some(v) => T::from_value(v).map_err(|e| format!("`{}.{field}`: {e}", self.name)),
            None => Err(format!(
                "annotation `{}` has no field `{field}`",
                self.name
            )),
        }
    }
}

/// Methods shared by annotated declarations.
macro_rules! annotated {
    ($($t:ty),*) => {$(
        impl $t {
            /// All annotations, in source order.
            pub fn all_annotations(&self) -> &[Annotation] {
                &self.annotations
            }

            /// The last `name` annotation (or `None`).
            pub fn annotation(&self, name: &str) -> Option<&Annotation> {
                self.annotations.iter().rev().find(|a| a.name == name)
            }

            /// All `name` annotations.
            pub fn annotations<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Annotation> {
                self.annotations.iter().filter(move |a| a.name == name)
            }

            /// Text of `@[Doc("…")]`.
            pub fn doc(&self) -> Option<&str> {
                match self.annotation("Doc")?.value("text")? {
                    Value::Str(s) => Some(s),
                    _ => None,
                }
            }

            /// `@[Deprecated(message?)]`: `Some(message)` if the declaration is deprecated.
            pub fn deprecated(&self) -> Option<Option<&str>> {
                let a = self.annotation("Deprecated")?;
                Some(match a.value("message") {
                    Some(Value::Str(s)) => Some(s),
                    _ => None,
                })
            }
        }
    )*};
}

/// Kind of declaration of a [`TypeDesc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclKind {
    Type,
    Class,
    Annotation,
    Trait,
}

/// A declared `type`, `class` or `annotation`.
#[derive(Clone, Debug)]
pub struct TypeDesc {
    pub name: String,
    pub kind: DeclKind,
    /// Parent class (`class B < A`).
    pub parent: Option<String>,
    /// Traits included (by the class or its parents).
    pub traits: Vec<String>,
    /// `abstract class`.
    pub is_abstract: bool,
    /// Fields, the parent's first.
    pub fields: Vec<FieldDesc>,
    /// Methods, in declaration order (inherited ones included).
    pub methods: Vec<FnDesc>,
    annotations: Vec<Annotation>,
}

impl TypeDesc {
    pub fn fields(&self) -> &[FieldDesc] {
        &self.fields
    }

    pub fn field(&self, name: &str) -> Option<&FieldDesc> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn methods(&self) -> &[FnDesc] {
        &self.methods
    }

    pub fn method(&self, name: &str) -> Option<&FnDesc> {
        self.methods.iter().find(|m| m.name == name)
    }
}

/// Field of a `type` / `class`.
#[derive(Clone, Debug)]
pub struct FieldDesc {
    pub name: String,
    /// Type annotation as written (`int`, `[Route]`…).
    pub ty: Option<String>,
    pub has_default: bool,
    annotations: Vec<Annotation>,
}

/// Function or method.
#[derive(Clone, Debug)]
pub struct FnDesc {
    pub name: String,
    pub params: Vec<ParamDesc>,
    pub ret: Option<String>,
    /// Method declared `private` or `protected`.
    pub visibility: Visibility,
    annotations: Vec<Annotation>,
}

impl FnDesc {
    pub fn param(&self, name: &str) -> Option<&ParamDesc> {
        self.params.iter().find(|p| p.name == name)
    }
}

/// Function parameter.
#[derive(Clone, Debug)]
pub struct ParamDesc {
    pub name: String,
    pub ty: Option<String>,
    pub has_default: bool,
    annotations: Vec<Annotation>,
}

/// A declared `enum`.
#[derive(Clone, Debug)]
pub struct EnumDesc {
    pub name: String,
    pub variants: Vec<VariantDesc>,
    pub methods: Vec<FnDesc>,
    annotations: Vec<Annotation>,
}

impl EnumDesc {
    pub fn variant(&self, name: &str) -> Option<&VariantDesc> {
        self.variants.iter().find(|v| v.name == name)
    }
}

/// Variant of an `enum`.
#[derive(Clone, Debug)]
pub struct VariantDesc {
    pub name: String,
    pub value: Value,
    annotations: Vec<Annotation>,
}

annotated!(
    TypeDesc,
    FieldDesc,
    FnDesc,
    ParamDesc,
    EnumDesc,
    VariantDesc
);

impl Runtime {
    /// Builds the annotations (evaluates their arguments).
    pub(crate) fn eval_annotations(&mut self, anns: &Annotations) -> R<Vec<Annotation>> {
        let mut out = Vec::with_capacity(anns.len());
        for a in anns.iter() {
            let ctx: Rc<[u32]> = Rc::from([]);
            let v = self.invoke(
                &a.make,
                &[],
                &ctx,
                None,
                CallArgs::plain(Vec::new()),
                a.span,
            )?;
            let Value::Object(o) = &v else {
                return type_err(
                    format!("`{}` is not an annotation", self.name(a.name)),
                    a.span,
                );
            };
            let o = o.borrow();
            let fields = o
                .info
                .fields
                .iter()
                .zip(&o.fields)
                .map(|(f, v)| (self.name(f.name).to_string(), v.clone()))
                .collect();
            out.push(Annotation {
                name: o.info.name.to_string(),
                fields,
            });
        }
        Ok(out)
    }

    pub(crate) fn fn_desc(&mut self, name: &str, p: &Proto) -> R<FnDesc> {
        let mut params = Vec::with_capacity(p.params.len());
        for prm in &p.params {
            params.push(ParamDesc {
                name: self.name(prm.name).to_string(),
                ty: prm.ty.as_ref().map(|t| t.shown.to_string()),
                has_default: prm.has_default,
                annotations: self.eval_annotations(&prm.annotations)?,
            });
        }
        Ok(FnDesc {
            name: name.to_string(),
            params,
            ret: p.ret.as_ref().map(|t| t.shown.to_string()),
            visibility: p.vis,
            annotations: self.eval_annotations(&p.annotations)?,
        })
    }

    /// Methods, in declaration order.
    fn methods_desc(
        &mut self,
        methods: &crate::intern::FxHashMap<crate::intern::Symbol, Rc<Proto>>,
    ) -> R<Vec<FnDesc>> {
        let mut ms: Vec<_> = methods.iter().map(|(n, p)| (*n, p.clone())).collect();
        ms.sort_by_key(|(_, p)| p.decl.as_ref().map_or(0, |d| d.span.start));
        let mut out = Vec::with_capacity(ms.len());
        for (n, p) in ms {
            let name = self.name(n).to_string();
            out.push(self.fn_desc(&name, &p)?);
        }
        Ok(out)
    }

    pub(crate) fn type_desc(&mut self, info: &TypeInfo) -> R<TypeDesc> {
        let mut fields = Vec::with_capacity(info.fields.len());
        for f in &info.fields {
            fields.push(FieldDesc {
                name: self.name(f.name).to_string(),
                ty: f.ty.as_ref().map(|t| t.shown.to_string()),
                has_default: f.default.is_some(),
                annotations: self.eval_annotations(&f.annotations)?,
            });
        }
        Ok(TypeDesc {
            name: info.name.to_string(),
            kind: match info.kind {
                crate::ast::TypeDeclKind::Type => DeclKind::Type,
                crate::ast::TypeDeclKind::Class => DeclKind::Class,
                crate::ast::TypeDeclKind::Annotation => DeclKind::Annotation,
                crate::ast::TypeDeclKind::Trait => DeclKind::Trait,
            },
            traits: info.traits.iter().map(|t| t.name.to_string()).collect(),
            is_abstract: info.is_abstract,
            parent: info.parent.as_ref().map(|p| p.name.to_string()),
            fields,
            methods: self.methods_desc(&info.methods)?,
            annotations: self.eval_annotations(&info.annotations)?,
        })
    }

    pub(crate) fn enum_desc(&mut self, info: &EnumInfo) -> R<EnumDesc> {
        let mut variants = Vec::with_capacity(info.variants.len());
        for (k, (n, v)) in info.variants.iter().enumerate() {
            let anns = info
                .variant_annotations
                .get(k)
                .cloned()
                .unwrap_or_else(|| Rc::new([]));
            variants.push(VariantDesc {
                name: self.name(*n).to_string(),
                value: v.clone(),
                annotations: self.eval_annotations(&anns)?,
            });
        }
        Ok(EnumDesc {
            name: info.name.to_string(),
            variants,
            methods: self.methods_desc(&info.methods)?,
            annotations: self.eval_annotations(&info.annotations)?,
        })
    }
}
