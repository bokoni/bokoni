# Syntaxe

Langage de script intégré à Rust, à la syntaxe inspirée de Ruby / Crystal, avec un typage graduel.

## Principes

- Syntaxe proche de Ruby : blocs `… end`, `#` pour les commentaires, pas d'accolades pour les blocs.
- Tout est une expression : la dernière valeur d'un bloc est sa valeur.
- Les appels de fonction prennent toujours des parenthèses : `print(x)`, pas `print x`.
- Fin de ligne = fin d'instruction. Une ligne qui se termine par un opérateur, `,` ou `(` continue sur la suivante.

## Typage graduel

Les annotations sont optionnelles. Quand elles sont présentes, elles sont vérifiées.

| Cas                          | Type                                  |
|------------------------------|---------------------------------------|
| `let x: int = 1`             | `int` (annoté)                        |
| `let x = 1`                  | `int` (déduit de la valeur)           |
| paramètre `a: int`           | `int`                                 |
| paramètre `a` sans type      | `any` (dynamique, vérifié à l'exécution) |

Une variable garde son type : `let x = 1` puis `x = "a"` est une erreur.

### Vérification statique

Avant l'exécution, le programme est vérifié en entier et **toutes** les erreurs sont
signalées d'un coup (syntaxe, puis noms et types) :

- noms : variable non définie, utilisée avant sa déclaration, assignée sans `let`,
  constante ou fonction modifiée, champ inconnu dans un bloc de construction ;
- types : `let` et réassignations, arguments (nombre, types, noms), retour, opérateurs,
  champs, méthodes, variantes d'enum, types inconnus dans les annotations ;
- inférence : type des variables, retour des fonctions non annotées, paramètres des
  lambdas passés à `map`, `filter`, `each`… (`[1, 2].map(|n| n.upper())` est refusé).

Un nom mal écrit est suggéré : `` `str` n'a pas de méthode `lenght` (voulez-vous dire `len` ?) ``.

**Seules les erreurs certaines sont signalées** (typage graduel) :

- une valeur de type inconnu (`any` : paramètre non annoté, fonction Rust) n'est jamais une
  erreur ;
- pour une valeur qui peut avoir plusieurs types (`int?`, `int | str`), une opération n'est
  refusée que si aucun de ces types ne la permet ;
- un `str` peut être passé là où `"dev" | "prod"` est attendu : la valeur exacte est
  vérifiée à l'exécution.

Ce qui n'est pas certain est vérifié à l'exécution. Depuis Rust, la vérification statique
peut être désactivée (`engine.set_static_checks(false)`).

### Types

| Type              | Exemple                       |
|-------------------|-------------------------------|
| `int`             | `42`, `-1`, `1_000`           |
| `float`           | `3.14`, `1e300`, `2.5e-8`     |
| `str`             | `"hello #{name}"`             |
| `bool`            | `true`, `false`               |
| `nil`             | `nil`                         |
| `any`             | n'importe quelle valeur       |
| `[T]`             | `[1, 2, 3]`                   |
| `{K => V}`        | hash : `{"a" => 1}`           |
| `{a: T, b: U}`    | record : `{a: 1, b: "x"}`     |
| `(A, B)`          | tuple : `(1, "x")`            |
| `T?`              | `str?` = `str \| nil`         |
| `A \| B`          | `int \| str`                  |
| littéraux         | `"dev" \| "prod"`, `2 \| 4`   |
| `(A, B): R`       | type d'une fonction / lambda  |

Un décimal s'affiche avec l'écriture la plus courte qui redonne exactement sa valeur :
`3.0`, `0.1`, `1e300`, `5e-324`, `1.2345678901234568e17` (notation scientifique pour les
très grands et très petits nombres), `inf`, `-inf`, `NaN`. Cette écriture se relit comme
littéral.

Les types primitifs sont en minuscules, les types utilisateur commencent par une majuscule.

## Variables

```ruby
let name = "Vincent"          # déclaration (type déduit : str)
let age: int = 42             # déclaration annotée
age = 43                      # réassignation d'une variable existante
MAX_PORT = 65535              # constante (MAJUSCULES, non réassignable)

total = 1                     # erreur : "total" n'est pas déclaré
```

- `let` déclare, `=` seul réassigne.
- Assigner une variable non déclarée est une erreur (évite les fautes de frappe silencieuses).
- Opérateurs composés : `+=`, `-=`, `*=`, `/=`, `<<` (ajout à une liste).

## Collections et structures

### Liste

```ruby
let nums = [1, 2, 3]          # [int]
let empty: [str] = []

nums[0]                       # 1
nums[-1]                      # 3 (depuis la fin)
nums << 4                     # ajout
nums.len()                    # 4
```

### Hash (clés dynamiques)

Un hash associe des clés à des valeurs, toutes du même type. Les clés sont des expressions, séparées par `=>`.

```ruby
let ages = {"alice" => 30, "bob" => 25}     # {str => int}
let names = {1 => "un", 2 => "deux"}        # {int => str}
let key = "carol"
let scores: {str => int} = {}

ages["alice"]                 # 30 : type int? (nil si la clé est absente)
ages["dave"] = 41             # ajout / modification
ages.has("bob")               # true
ages.delete("bob")
ages.keys()                   # ["alice", "dave"]
scores[key] = 10              # clé calculée

for name, age in ages
  print("#{name} a #{age} ans")
end
```

### Record (structure anonyme)

Un record a un ensemble fixe de champs, chacun avec son propre type. Les noms de champs sont écrits directement, suivis de `:`.

```ruby
let point = {x: 1, y: 2}                    # {x: int, y: int}
let user = {name: "Vincent", age: 42, tags: ["admin"]}

point.x                       # 1
point.x = 5                   # modification d'un champ existant
point.z = 3                   # erreur : le record n'a pas de champ "z"

let x = 10
let p2 = {x, y: 20}           # raccourci : {x: x, y: 20}

def dist(p: {x: int, y: int}): float
  (p.x * p.x + p.y * p.y).sqrt()
end
```

Un `type` est un record nommé, avec des valeurs par défaut. Un record anonyme sert pour les valeurs ponctuelles (retours multiples, options, données passées à une fonction).

### Compatibilité record / type

Les `type` et les `class` sont **nominaux** : deux types différents ne sont jamais interchangeables, même s'ils ont les mêmes champs.

```ruby
type Euro
  montant: float
end

type Dollar
  montant: float
end

def payer(e: Euro) … end

payer(Dollar(montant: 10.0))  # erreur : Dollar n'est pas Euro
```

Exception : un **record littéral** écrit directement là où un type est attendu est converti vers ce type. Les champs sont vérifiés, et les valeurs par défaut s'appliquent.

```ruby
def register(r: Route) … end

register({path: "/users"})               # OK : converti en Route (methods = ["GET"])
let s: Server = {host: "api", port: 443} # OK : idem pour une annotation
register({path: "/x", verb: "GET"})      # erreur : Route n'a pas de champ "verb"

let r = {path: "/users"}
register(r)                              # erreur : r est un record, pas un Route
register(Route(path: r.path))            # OK
```

La conversion a lieu partout où le type attendu est connu : argument de fonction, `let x: T = …`, valeur de retour d'une fonction `: T`, élément d'une liste `[T]`.

### Hash ou record ?

|                   | Hash `{k => v}`                    | Record `{k: v}`                  |
|-------------------|------------------------------------|----------------------------------|
| clés              | n'importe quelle expression        | noms fixes, connus à l'écriture  |
| types des valeurs | tous identiques (`V`)              | un type par champ                |
| accès             | `h["clé"]` (peut être `nil`)       | `r.champ` (vérifié)              |
| ajout de clé      | oui                                | non                              |
| usage             | dictionnaire, index, cache         | structure, configuration         |

### Tuple et déstructuration

```ruby
let pair = (1, "un")          # (int, str)
pair.0                        # 1

def min_max(nums: [int]): (int, int)
  (nums.min(), nums.max())
end

let (lo, hi) = min_max([3, 1, 4])
let {x, y} = point            # déstructuration d'un record
let [first, ...rest] = nums   # déstructuration d'une liste
```

## Fonctions

```ruby
def add(a: int, b: int = 0): int
  a + b
end

def greet(who)                # paramètre non typé → any, retour déduit
  print("Bonjour #{who}")
end

def double(x: int): int = x * 2   # forme courte sur une ligne

add(1, 2)
add(a: 1, b: 2)               # arguments nommés
```

- Le type de retour s'écrit `: type` après les paramètres.
- `return` existe pour sortir plus tôt ; sinon la dernière expression est retournée.

## Lambdas et blocs

```ruby
let inc = |x| x + 1
let add = |a: int, b: int| a + b
let hello = || print("hello")

[1, 2, 3].map(|n| n * 2)

[1, 2, 3].each do |n|
  print(n)
end
```

`do |params| … end` passe un bloc en dernier argument à l'appel qui le précède.

## Contrôle

```ruby
let level = if age >= 18
  "adulte"
elsif age >= 12
  "ado"
else
  "enfant"
end

let short = if ok then "oui" else "non" end

let kind = case port
  when 80, 443 then "web"
  when 8000..8999 then "dev"
  else "autre"
end

while age < 50
  age += 1
end

for t in tags
  print(t)
end

for i in 0..10                # 0..10 inclusif, 0...10 exclusif
  next if i % 2 == 0
  break if i > 7
  print(i)
end
```

Modificateurs en fin de ligne : `expr if cond`, `expr unless cond`.

## Enum

```ruby
enum Color
  Red
  Green
  Blue
end

enum Protocol: str
  Tcp = "TCP"
  Udp = "UDP"
end

let p = Protocol::Tcp
p.value                       # "TCP"
p.name                        # "Tcp"
Color::Green.value            # 1 (sans valeur : l'index ; pour un enum `: str`, le nom)

Protocol.values()             # [Protocol::Tcp, Protocol::Udp]
Protocol.from("UDP")          # Protocol::Udp (nil si aucune variante n'a cette valeur)

enum Level: int
  Low = 1
  High = 10

  def is_high: bool = self == Level::High   # méthode, `self` = la variante
end
```

## Type (données)

Les déclarations `type`, `class`, `annotation` et `enum` se font au niveau principal du
script (pas dans une fonction ou un bloc). Elles sont visibles dans tout le script, même
avant leur déclaration.

Un `type` décrit une structure de données : des champs, avec types et valeurs par défaut.
Un champ par défaut peut dépendre des champs précédents.

```ruby
type Route
  path: str
  methods: [str] = ["GET"]
end

type Server
  host: str = "localhost"
  protocol = Protocol::Tcp
  port: int = if protocol == Protocol::Tcp then 80 else 40 end
  debug: bool = false
  mode: "dev" | "prod" = "dev"
  routes: [Route] = []
end
```

Un champ sans valeur par défaut (`path: str`) est obligatoire à la construction.

### Construction

Forme courte, avec des arguments nommés :

```ruby
let r = Route(path: "/health")
```

Forme bloc, pour la configuration :

```ruby
let env = "prod"

let s = Server do
  let base = 8000             # variable locale au bloc
  host = "api.example.com"    # remplit le champ "host"
  mode = env                  # lit une variable extérieure
  port = if mode == "prod" then 443 else base + 80 end
  debug = mode == "dev"

  routes << Route do
    path = "/users"
    methods = ["GET", "POST"]
  end

  for p in ["/health", "/metrics"]
    routes << Route(path: p)
  end

  prot = 443                  # erreur : Server n'a pas de champ "prot"
end
```

Règles dans un bloc de construction :

- `champ = valeur` remplit un champ du type.
- `let x = …` déclare une variable locale au bloc.
- Un nom non déclaré qui n'est pas un champ est une erreur.
- Un champ lu dans le bloc (`routes << …`) reçoit d'abord sa valeur par défaut.
- Les champs jamais touchés prennent leur valeur par défaut **après** le bloc : elle voit
  les champs remplis par le bloc (`protocol = Protocol::Udp` donne `port = 40`).
- Une dépendance circulaire entre valeurs par défaut est une erreur.

Les champs d'un `type` sont publics (`s.port`, `s.port = 8080`, avec vérification du type).
Deux valeurs d'un même `type` sont égales si tous leurs champs le sont.

## Classe (objets)

Une `class` porte un état et des méthodes. `@champ` désigne un attribut de l'instance.

```ruby
class User
  # @name et @email sont déclarés et assignés directement par le constructeur
  def init(@name: str, @email: str? = nil)
  end

  def name: str = @name                 # getter

  def name=(value: str)                 # setter
    @name = value.capitalize()
  end

  def to_s: str = "User(#{@name})"
end

let u = User("vincent")
u.name = "bob"                           # appelle name=
print(u.name)
```

- `init` est le seul constructeur ; un paramètre `@x` déclare et assigne le champ.
- Les champs sont privés : lus et modifiés par `@x` dans les méthodes ; de l'extérieur,
  uniquement par un getter (`def name`) ou un setter (`def name=(v)`).
- Un champ peut aussi être déclaré dans le corps, avec une valeur par défaut
  (`count = 0`) ; tout champ doit être initialisé à la fin de `init`.
- `self` désigne l'instance. `to_s` est utilisé par `print` et l'interpolation.
- Deux instances ne sont égales que si c'est le même objet.
- Pas de destructeur.

## Commentaires

```ruby
# commentaire sur une ligne
```

## Annotations

> Inspiré de [Crystal](https://crystal-lang.org/reference/1.22/syntax_and_semantics/annotations/index.html).

Une annotation attache des métadonnées à une déclaration. Elle ne change rien à l'exécution par elle-même : elle est lue par le code Rust hôte (et, plus tard, par les macros).

### Déclaration

Une annotation se déclare comme un `type` : ses arguments sont donc typés et vérifiés.

```ruby
annotation Validate
  min: int? = nil
  max: int? = nil
end

annotation Column
  name: str? = nil
  primary: bool = false
end

annotation Tag                # sans argument
end
```

### Application

`@[Nom]` ou `@[Nom(args)]`, placé avant la déclaration. Aucun conflit avec `@champ` : `@[` est un jeton distinct.

```ruby
@[Tag]
type Server
  @[Validate(min: 1, max: 65535)]
  port: int = 80

  @[Column(name: "host_name")]
  @[Deprecated("utiliser address")]
  host: str = "localhost"
end

@[Deprecated("utiliser add")]
def plus(a: int, b: int): int = a + b

def connect(@[Validate(min: 0)] timeout: int)
end

enum Protocol: str
  @[Deprecated]
  Ftp = "FTP"
  Tcp = "TCP"
end
```

- Cibles : `type`, `class`, `enum`, variante d'enum, champ, méthode, fonction, paramètre.
- Arguments nommés (`min: 1`) ou positionnels dans l'ordre des champs (`Validate(1, 65535)`).
- Plusieurs annotations, y compris du même type, sur une même cible.
- Les arguments sont vérifiés avant l'exécution, comme une construction : annotation
  inconnue, champ inexistant, type d'argument, champ obligatoire manquant.
- Ils sont évalués **à la lecture** (depuis Rust), pas à la déclaration ; ce sont des
  constantes : ils peuvent utiliser des globales (`@[Limit(MAX + 2)]`), pas des variables
  locales.

### Lecture

Dans une macro (prévu : version 3) :

```ruby
macro validate_fields
  def validate(): [str]
    let errors: [str] = []
    {% for f in @type.fields %}
      {% if v = f.annotation(Validate) %}
        {% if v.min %}
          errors << "{{f.name}} < {{v.min}}" if @{{f.name}} < {{v.min}}
        {% end %}
      {% end %}
    {% end %}
    errors
  end
end
```

- `x.annotation(A)` : la dernière annotation `A` (ou `nil`).
- `x.annotations(A)` / `x.annotations()` : toutes les annotations `A` / toutes.
- Les arguments se lisent comme des champs : `v.min`, `v.max`.

Depuis Rust :

```rust
let ty = engine.type_info("Server")?;          // aussi : function_info, enum_info
for field in ty.fields() {
    if let Some(v) = field.annotation("Validate") {
        let min: Option<i64> = v.get("min")?;
    }
}
let doc = ty.doc();                              // texte de @[Doc("…")]
let obsolete = ty.field("host").unwrap().deprecated(); // Some(Some("utiliser address"))
```

`type_info`, `function_info` et `enum_info` renvoient une description (champs avec leur
type, méthodes, paramètres, variantes avec leur valeur) qui ne dépend plus du moteur. Sur
chaque élément : `annotation(nom)` (la dernière), `annotations(nom)`, `all_annotations()`,
`doc()`, `deprecated()` ; sur une annotation : `name()`, `fields()`, `get::<T>(champ)`.

### Annotations intégrées

| Annotation             | Effet                                                   |
|------------------------|---------------------------------------------------------|
| `@[Deprecated(msg?)]`  | avertissement à chaque utilisation                      |
| `@[Doc("…")]`          | documentation (alternative aux commentaires)            |

Elles sont déclarées par le moteur comme des annotations ordinaires
(`Deprecated(message: str? = nil)`, `Doc(text: str)`).

L'avertissement de `Deprecated` est émis par la vérification, avant l'exécution, à chaque
utilisation : appel ou mention d'une fonction, d'un type, d'un enum, d'une variante
(`Protocol::Ftp`), lecture ou passage par nom d'un champ, appel d'une méthode — y compris
depuis un script suivant. Depuis Rust : `engine.take_warnings()` ou
`engine.set_warning_handler(…)` ; l'exécutable les affiche sur la sortie d'erreur.

```text
avertissement: `plus` est obsolète : utiliser add
 --> script.bk:17:7
   |
17 | print(plus(1, 2))
   |       ^^^^
```

## Macros (prévu : version 3)

> Inspiré de [Crystal](https://crystal-lang.org/reference/1.22/syntax_and_semantics/macros/index.html).

Une macro s'exécute **à la compilation** : elle reçoit des nœuds de l'AST (pas des valeurs) et génère du code qui est inséré à sa place.

- `{{ expr }}` : insère un nœud / une valeur dans le code généré.
- `{% … %}` : logique exécutée à la compilation (`if`, `for`, `let`).

```ruby
macro getters(*names)
  {% for name in names %}
    def {{name}} = @{{name}}
  {% end %}
end

class User
  def init(@name: str, @email: str)
  end

  getters(name, email)        # génère : def name = @name / def email = @email
end
```

```ruby
macro derive_to_s
  def to_s: str
    {% let parts = @type.fields.map(|f| "#{f.name}=\#{@#{f.name}}") %}
    "{{@type.name}}({{parts.join(", ")}})"
  end
end

type Point
  x: int = 0
  y: int = 0
  derive_to_s()
end

Point(x: 1).to_s()            # "Point(x=1, y=0)"
```

### Contexte disponible dans une macro

| Variable     | Contenu                                               |
|--------------|-------------------------------------------------------|
| `@type`      | le type courant : `name`, `fields`, `methods`, `annotation(…)` |
| `@def`       | la méthode courante : `name`, `params`, `return_type`, `annotation(…)` |
| `name.id`    | un nœud converti en identifiant                       |
| `node.stringify` | le code source d'un nœud, en `str`               |

### Règles

- Le code généré doit être valide à lui seul (pas de `when` isolé hors d'un `case`).
- Hygiène : les variables déclarées par `let` dans le code généré ne fuient pas chez l'appelant (renommage automatique).
- Les erreurs pointent à la fois vers l'appel de la macro et vers la ligne générée fautive.
- Une macro est visible dans le fichier / le type où elle est définie, et dans ses descendants.

### Prérequis dès la version 1

- AST complet avec positions source (fichier, ligne, colonne) sur chaque nœud.
- AST sérialisable et affichable en code source (`stringify`).
- Le parseur accepte et conserve `@[…]` (fait : les annotations sont lues depuis Rust).

## Mots réservés

`let def end if elsif else unless then case when while for in do break next return
type class enum annotation macro true false nil and or not`

## Points encore ouverts

- Héritage / interfaces pour `class` (et `type` ?).
- Modules / import entre fichiers.
- Gestion des erreurs (`raise` / `rescue` ou `Result`).
- Méthodes sur `type` (autorisées, ou réservées à `class`).
- Généricité (`[T]`, `fn<T>`).