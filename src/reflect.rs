//! Introspection depuis Rust : déclarations du script et leurs annotations.
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
//! Les descriptions sont des copies : elles ne gardent aucun lien avec le moteur. Les
//! arguments des annotations sont évalués au moment de la lecture.

use crate::engine::FromValue;
use crate::ir::{Annotations, Proto};
use crate::objects::{EnumInfo, TypeInfo};
use crate::ops::{R, type_err};
use crate::value::Value;
use crate::vm::{CallArgs, Runtime};
use std::rc::Rc;

/// Annotation appliquée (`@[Validate(min: 1)]`) : son nom et la valeur de ses champs.
#[derive(Clone, Debug)]
pub struct Annotation {
    name: String,
    fields: Vec<(String, Value)>,
}

impl Annotation {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Champs de l'annotation, dans l'ordre de sa déclaration (valeurs par défaut
    /// comprises).
    pub fn fields(&self) -> &[(String, Value)] {
        &self.fields
    }

    /// Valeur d'un champ.
    pub fn value(&self, field: &str) -> Option<&Value> {
        self.fields.iter().find(|(n, _)| n == field).map(|(_, v)| v)
    }

    /// Valeur d'un champ, convertie (`v.get::<Option<i64>>("min")`).
    pub fn get<T: FromValue>(&self, field: &str) -> Result<T, String> {
        match self.value(field) {
            Some(v) => T::from_value(v).map_err(|e| format!("`{}.{field}` : {e}", self.name)),
            None => Err(format!(
                "l'annotation `{}` n'a pas de champ `{field}`",
                self.name
            )),
        }
    }
}

/// Méthodes communes aux déclarations annotées.
macro_rules! annotated {
    ($($t:ty),*) => {$(
        impl $t {
            /// Toutes les annotations, dans l'ordre d'écriture.
            pub fn all_annotations(&self) -> &[Annotation] {
                &self.annotations
            }

            /// La dernière annotation `name` (ou `None`).
            pub fn annotation(&self, name: &str) -> Option<&Annotation> {
                self.annotations.iter().rev().find(|a| a.name == name)
            }

            /// Toutes les annotations `name`.
            pub fn annotations<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Annotation> {
                self.annotations.iter().filter(move |a| a.name == name)
            }

            /// Texte de `@[Doc("…")]`.
            pub fn doc(&self) -> Option<&str> {
                match self.annotation("Doc")?.value("text")? {
                    Value::Str(s) => Some(s),
                    _ => None,
                }
            }

            /// `@[Deprecated(message?)]` : `Some(message)` si la déclaration est obsolète.
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

/// Sorte de déclaration de [`TypeDesc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclKind {
    Type,
    Class,
    Annotation,
}

/// `type`, `class` ou `annotation` déclaré.
#[derive(Clone, Debug)]
pub struct TypeDesc {
    pub name: String,
    pub kind: DeclKind,
    pub fields: Vec<FieldDesc>,
    /// Méthodes, dans l'ordre de leur déclaration.
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

/// Champ d'un `type` / d'une `class`.
#[derive(Clone, Debug)]
pub struct FieldDesc {
    pub name: String,
    /// Annotation de type telle qu'écrite (`int`, `[Route]`…).
    pub ty: Option<String>,
    pub has_default: bool,
    annotations: Vec<Annotation>,
}

/// Fonction ou méthode.
#[derive(Clone, Debug)]
pub struct FnDesc {
    pub name: String,
    pub params: Vec<ParamDesc>,
    pub ret: Option<String>,
    annotations: Vec<Annotation>,
}

impl FnDesc {
    pub fn param(&self, name: &str) -> Option<&ParamDesc> {
        self.params.iter().find(|p| p.name == name)
    }
}

/// Paramètre d'une fonction.
#[derive(Clone, Debug)]
pub struct ParamDesc {
    pub name: String,
    pub ty: Option<String>,
    pub has_default: bool,
    annotations: Vec<Annotation>,
}

/// `enum` déclaré.
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

/// Variante d'un `enum`.
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
    /// Construit les annotations (évalue leurs arguments).
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
                    format!("`{}` n'est pas une annotation", self.name(a.name)),
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
            annotations: self.eval_annotations(&p.annotations)?,
        })
    }

    /// Méthodes, dans l'ordre de leur déclaration.
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
            },
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
