# Syntax

A scripting language embedded in Rust, with syntax inspired by Ruby / Crystal and gradual typing.

## Principles

- Syntax close to Ruby: `… end` blocks, `#` for comments, no braces for blocks.
- Everything is an expression: the last value of a block is its value.
- Function calls always take parentheses: `print(x)`, not `print x`.
- End of line = end of statement. A line ending with an operator, `,` or `(` continues on the next one.

## Gradual typing

Annotations are optional. When present, they are checked.

| Case                         | Type                                  |
|------------------------------|---------------------------------------|
| `let x: int = 1`             | `int` (annotated)                     |
| `let x = 1`                  | `int` (inferred from the value)       |
| parameter `a: int`           | `int`                                 |
| parameter `a` without a type | `any` (dynamic, checked at run time)  |

A variable keeps its type: `let x = 1` followed by `x = "a"` is an error.

### Static checking

Before running, the whole program is checked and **all** errors are reported at once
(syntax, then names and types):

- names: undefined variable, used before its declaration, assigned without `let`,
  constant or function modified, unknown field in a construction block;
- types: `let` and reassignments, arguments (count, types, names), return, operators,
  fields, methods, enum variants, unknown types in annotations;
- inference: variable types, return type of unannotated functions, parameters of lambdas
  passed to `map`, `filter`, `each`… (`[1, 2].map(|n| n.upper())` is rejected).

A misspelled name gets a suggestion: `` `str` has no method `lenght` (did you mean `len`?) ``.

**Only certain errors are reported** (gradual typing):

- a value of unknown type (`any`: unannotated parameter, Rust function) is never an
  error;
- for a value that may have several types (`int?`, `int | str`), an operation is rejected
  only if none of those types allows it;
- a `str` may be passed where `"dev" | "prod"` is expected: the exact value is checked at
  run time.

Whatever is not certain is checked at run time. From Rust, static checking can be
disabled (`engine.set_static_checks(false)`).

### Types

| Type              | Example                       |
|-------------------|-------------------------------|
| `int`             | `42`, `-1`, `1_000`           |
| `float`           | `3.14`, `1e300`, `2.5e-8`     |
| `str`             | `"hello #{name}"`             |
| `bool`            | `true`, `false`               |
| `nil`             | `nil`                         |
| `any`             | any value                     |
| `[T]`             | `[1, 2, 3]`                   |
| `{K => V}`        | hash: `{"a" => 1}`            |
| `{a: T, b: U}`    | record: `{a: 1, b: "x"}`      |
| `(A, B)`          | tuple: `(1, "x")`             |
| `T?`              | `str?` = `str \| nil`         |
| `A \| B`          | `int \| str`                  |
| literals          | `"dev" \| "prod"`, `2 \| 4`   |
| `(A, B): R`       | type of a function / lambda   |

A float is displayed with the shortest form that gives back exactly its value: `3.0`,
`0.1`, `1e300`, `5e-324`, `1.2345678901234568e17` (scientific notation for very large and
very small numbers), `inf`, `-inf`, `NaN`. That form reads back as a literal.

Primitive types are lowercase; user types start with an uppercase letter.

## Variables

```ruby
let name = "Vincent"          # declaration (inferred type: str)
let age: int = 42             # annotated declaration
age = 43                      # reassignment of an existing variable
MAX_PORT = 65535              # constant (UPPERCASE, cannot be reassigned)

total = 1                     # error: "total" is not declared
```

- `let` declares, `=` alone reassigns.
- Assigning an undeclared variable is an error (avoids silent typos).
- Compound operators: `+=`, `-=`, `*=`, `/=`, `<<` (append to a list).

## Collections and structures

### List

```ruby
let nums = [1, 2, 3]          # [int]
let empty: [str] = []

nums[0]                       # 1
nums[-1]                      # 3 (from the end)
nums << 4                     # append
nums.len()                    # 4
```

### Hash (dynamic keys)

A hash maps keys to values, all of the same type. Keys are expressions, separated by `=>`.

```ruby
let ages = {"alice" => 30, "bob" => 25}     # {str => int}
let names = {1 => "one", 2 => "two"}        # {int => str}
let key = "carol"
let scores: {str => int} = {}

ages["alice"]                 # 30: type int? (nil if the key is missing)
ages["dave"] = 41             # insert / update
ages.has("bob")               # true
ages.delete("bob")
ages.keys()                   # ["alice", "dave"]
scores[key] = 10              # computed key

for name, age in ages
  print("#{name} is #{age}")
end
```

### Record (anonymous structure)

A record has a fixed set of fields, each with its own type. Field names are written directly, followed by `:`.

```ruby
let point = {x: 1, y: 2}                    # {x: int, y: int}
let user = {name: "Vincent", age: 42, tags: ["admin"]}

point.x                       # 1
point.x = 5                   # update of an existing field
point.z = 3                   # error: the record has no field "z"

let x = 10
let p2 = {x, y: 20}           # shorthand: {x: x, y: 20}

def dist(p: {x: int, y: int}): float
  (p.x * p.x + p.y * p.y).sqrt()
end
```

A `type` is a named record, with default values. An anonymous record is for one-off values (multiple returns, options, data passed to a function).

### Record / type compatibility

`type` and `class` are **nominal**: two different types are never interchangeable, even if they have the same fields.

```ruby
type Euro
  amount: float
end

type Dollar
  amount: float
end

def pay(e: Euro) … end

pay(Dollar(amount: 10.0))     # error: Dollar is not Euro
```

Exception: a **record literal** written directly where a type is expected is converted to that type. Fields are checked, and default values apply.

```ruby
def register(r: Route) … end

register({path: "/users"})               # OK: converted to Route (methods = ["GET"])
let s: Server = {host: "api", port: 443} # OK: same for an annotation
register({path: "/x", verb: "GET"})      # error: Route has no field "verb"

let r = {path: "/users"}
register(r)                              # error: r is a record, not a Route
register(Route(path: r.path))            # OK
```

The conversion happens wherever the expected type is known: function argument, `let x: T = …`, return value of a `: T` function, element of a `[T]` list.

### Hash or record?

|                   | Hash `{k => v}`                    | Record `{k: v}`                  |
|-------------------|------------------------------------|----------------------------------|
| keys              | any expression                     | fixed names, known when written  |
| value types       | all the same (`V`)                 | one type per field               |
| access            | `h["key"]` (may be `nil`)          | `r.field` (checked)              |
| adding a key      | yes                                | no                               |
| use               | dictionary, index, cache           | structure, configuration         |

### Tuple and destructuring

```ruby
let pair = (1, "one")         # (int, str)
pair.0                        # 1

def min_max(nums: [int]): (int, int)
  (nums.min(), nums.max())
end

let (lo, hi) = min_max([3, 1, 4])
let {x, y} = point            # destructuring a record
let [first, ...rest] = nums   # destructuring a list
```

## Functions

```ruby
def add(a: int, b: int = 0): int
  a + b
end

def greet(who)                # untyped parameter → any, inferred return
  print("Hello #{who}")
end

def double(x: int): int = x * 2   # short one-line form

add(1, 2)
add(a: 1, b: 2)               # named arguments
```

- The return type is written `: type` after the parameters.
- `return` exists to exit early; otherwise the last expression is returned.

## Lambdas and blocks

```ruby
let inc = |x| x + 1
let add = |a: int, b: int| a + b
let hello = || print("hello")

[1, 2, 3].map(|n| n * 2)

[1, 2, 3].each do |n|
  print(n)
end
```

`do |params| … end` passes a block as the last argument of the call before it.

## Control flow

```ruby
let level = if age >= 18
  "adult"
elsif age >= 12
  "teen"
else
  "child"
end

let short = if ok then "yes" else "no" end

let kind = case port
  when 80, 443 then "web"
  when 8000..8999 then "dev"
  else "other"
end

while age < 50
  age += 1
end

for t in tags
  print(t)
end

for i in 0..10                # 0..10 inclusive, 0...10 exclusive
  next if i % 2 == 0
  break if i > 7
  print(i)
end
```

End-of-line modifiers: `expr if cond`, `expr unless cond`.

## Error handling

> Modeled on [Crystal](https://crystal-lang.org/reference/1.22/syntax_and_semantics/exception_handling.html): exceptions, `raise`, `begin` / `rescue` / `else` / `ensure`.

```ruby
class ConfigError < Error
  def init(@path: str)
    super("missing #{path}")
  end
  def path = @path
end

def load(path: str): str
  raise ConfigError(path) unless exists(path)
  read(path)
end

let text = begin
  load("app.yml")
rescue e: ConfigError          # by class (`A | B` for several)
  print("config: #{e.message} (#{e.path})")
  ""
rescue e: Error                # any other error: division by zero, index…
  print(e.message)
  ""
else                           # no error in the body
  print("loaded")
  ""
ensure                         # always: after the body, a `rescue`, a `return`…
  print("done")
end
```

- `raise "message"` raises an `Error`, a class of the prelude (`Error("message")`,
  `e.message`); `raise obj` raises an object of any `type` / `class` as is. Any other
  value is a type error. Custom errors inherit from `Error`: `rescue e: Error` then
  catches them too.
- Runtime errors (`1 / 0`, an index out of bounds, a type error at run time, too many
  nested calls…) can be rescued as an `Error`. Syntax errors and errors reported by the
  static checks cannot: the script does not run.
- `rescue`, `rescue e`, `rescue e: T`, `rescue T`: clauses are tried in order; a name
  that starts with an uppercase letter is a type. With no clause matching, the error
  goes on up unchanged (position and call stack included). A clause for a class also
  catches its subclasses: put the most specific first.
- The value of a `begin` is the value of its body (or of the `else`), or that of the
  `rescue` clause that ran; the value of `ensure` is ignored.
- `ensure` also runs when `return`, `break` or `next` leaves the `begin`, and when the
  error goes on up. An error raised inside `ensure` replaces the first one.
- The body of a `def` or of a `do` block can take `rescue` / `else` / `ensure` directly,
  without `begin`:

```ruby
def parse_port(s: str): int
  s.to_i()
rescue
  80
end

let ratios = [4, 0, 2].map do |x|
  100 / x
rescue
  nil
end                            # [25, nil, 50]
```

- An uncaught error stops the script: `runtime error: ConfigError: missing app.yml`
  (the `message` field of the object, if it has one).

### `?` variants: `nil` instead of an error

For a failure that is expected (input to validate), a `?` variant returns `nil` instead
of raising: cheaper than an exception, and the type (`int?`) says that the case must be
handled.

```ruby
"8080".to_i?()                 # 8080
"80x".to_i?()                  # nil   ("80x".to_i() raises an error)
"1.5".to_f?()                  # 1.5
1e300.to_i?()                  # nil   (too large for an `int`)
xs[10]?                        # nil when out of bounds (also for a `str`, a tuple)

let port = s.to_i?() or 80
```

A wrong type is still an error (`xs["a"]?`). The other built-in methods already return
`nil` when there is nothing to return: `first()`, `last()`, `pop()`, `min()`, `max()`,
`find()`, `h[key]`, `get(key)`.

The name of a function or a method can end with `?` (`def empty?() = …`,
`valid?(s)`), as in Ruby and Crystal.

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
Color::Green.value            # 1 (without a value: the index; for a `: str` enum, the name)

Protocol.values()             # [Protocol::Tcp, Protocol::Udp]
Protocol.from("UDP")          # Protocol::Udp (nil if no variant has that value)

enum Level: int
  Low = 1
  High = 10

  def is_high: bool = self == Level::High   # method, `self` = the variant
end
```

## Type (data)

`type`, `class`, `annotation` and `enum` declarations are made at the top level of the
script (not inside a function or a block). They are visible in the whole script, even
before their declaration.

A `type` describes a data structure: fields, with types and default values. A field's
default may depend on the previous fields.

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

A field without a default value (`path: str`) is required at construction.

### Construction

Short form, with named arguments:

```ruby
let r = Route(path: "/health")
```

Block form, for configuration:

```ruby
let env = "prod"

let s = Server do
  let base = 8000             # variable local to the block
  host = "api.example.com"    # fills the "host" field
  mode = env                  # reads an outer variable
  port = if mode == "prod" then 443 else base + 80 end
  debug = mode == "dev"

  routes << Route do
    path = "/users"
    methods = ["GET", "POST"]
  end

  for p in ["/health", "/metrics"]
    routes << Route(path: p)
  end

  prot = 443                  # error: Server has no field "prot"
end
```

Rules in a construction block:

- `field = value` fills a field of the type.
- `let x = …` declares a variable local to the block.
- An undeclared name that is not a field is an error.
- A field read in the block (`routes << …`) first receives its default value.
- Fields never touched take their default value **after** the block: it sees the fields
  filled by the block (`protocol = Protocol::Udp` gives `port = 40`).
- A circular dependency between default values is an error.

The fields of a `type` are public (`s.port`, `s.port = 8080`, with type checking). Two
values of the same `type` are equal if all their fields are.

## Class (objects)

A `class` carries state and methods. `@field` denotes an attribute of the instance.

```ruby
class User
  # @name and @email are declared and assigned directly by the constructor
  def init(@name: str, @email: str? = nil)
  end

  def name: str = @name                 # getter

  def name=(value: str)                 # setter
    @name = value.capitalize()
  end

  def to_s: str = "User(#{@name})"
end

let u = User("vincent")
u.name = "bob"                           # calls name=
print(u.name)
```

- `init` is the only constructor; an `@x` parameter declares and assigns the field.
- Fields are private: read and modified through `@x` in methods; from outside, only
  through a getter (`def name`) or a setter (`def name=(v)`).
- A field may also be declared in the body, with a default value (`count = 0`); every
  field must be initialized by the end of `init`.
- `self` denotes the instance. `to_s` is used by `print` and interpolation.
- Two instances are equal only if they are the same object.
- No destructor.

### Inheritance

Single inheritance, for classes only (a `type` describes data and does not inherit):

```ruby
class Animal
  def init(@name: str)
  end
  def speak(): str = "..."
  def describe(): str = "#{@name} says #{self.speak()}"
end

class Dog < Animal
  def speak(): str = "woof"                 # redefined
end

class Puppy < Dog
  def init(@name: str, @age: int)          # `@name`: the field of `Animal`
  end
  def speak(): str = super() + " (#{@age})" # `Dog`'s version
end

let pets: [Animal] = [Dog("rex"), Puppy("bit", 1)]
pets.map(|p| p.describe())                 # ["rex says woof", "bit says woof (1)"]
```

- The subclass has the parent's fields (first) and methods, `init` included; it can
  redefine a method, and add fields (not redeclare one of the parent's).
- `super(args)` calls the parent's version of the current method, in any method
  (`init` included), and also inside a `do` block of the method.
- A subclass instance is accepted wherever the parent is expected (`a: Animal`,
  `[Animal]`, `rescue e: Animal`); the reverse is a type error.
- The parent must be declared before, in the same script or a previous one. No multiple
  inheritance: share behavior with traits.

### Traits

A `trait` describes behavior shared by classes that are not related: methods to provide
(`abstract def`, a signature without body) and methods provided (with a body, which
can call the others through `self`). A class takes them with `include`, as many as it
needs:

```ruby
trait Greet
  abstract def name(): str
  def greet(): str = "Hello #{self.name()}"
end

class User
  include Greet
  def init(@name: str)
  end
  def name(): str = @name
end

class Robot
  include Greet
  def name(): str = "R2"
  def greet(): str = "beep"                 # redefines the trait's
end

def welcome(x: Greet): str = x.greet()      # a trait is a type
welcome(User("ann"))                        # "Hello ann"
```

- Precedence: the class's own methods, then the included traits' (in order), then the
  parent's.
- A class that does not define a required method is an error (before running):
  `` `User` must define `name` (required by `Greet`) ``. A method provided by another
  trait or by the parent counts.
- A trait has no fields and is not instantiated; it is nominal: only the classes that
  `include` it are of its type (`x: Greet`, `[Greet]`, `rescue e: Retryable`).
- `include` only in a class; a trait does not include another one (yet).

### Abstract classes

```ruby
abstract class Shape
  abstract def area(): float
  def describe(): str = "area #{self.area()}"
end

class Square < Shape
  def init(@side: float)
  end
  def area(): float = @side * @side
end

Shape()                                     # error: an abstract class
```

An `abstract class` cannot be instantiated and may declare `abstract def`s; its
subclasses must define them, unless they are abstract too.

### `private` and `protected` methods

```ruby
class Account
  def init(@cents: int)
  end

  private def valid(n: int): bool = n > 0
  protected def cents = @cents

  def deposit(n: int)
    raise Error("invalid amount") unless self.valid(n)   # on `self`: allowed
    @cents += n
  end

  def richer(o: Account): bool = self.cents > o.cents    # another Account: allowed
end

let a = Account(10)
a.deposit(5)
a.valid(1)          # error: `valid` is a private method of `Account`
a.cents             # error: `cents` is a protected method of `Account`
```

As in Ruby and Crystal:

- A `private` method is called on `self` only (`self.valid(n)`), in the methods of
  its class and of its subclasses; not on another object, even of the same class.
- A `protected` method is also called on other objects, but only in the methods of
  the class (or trait) that declares it and of its subclasses (or of the classes that
  include the trait).
- This applies to getters and setters too (`private def name=(v)`), and to the
  methods of a trait or a `type`. `init` stays public.
- The checker reports a misuse before running; without static checks, the same error
  happens when running.

## Generics

Functions and types may take type parameters, as in Crystal. They are checked by the
checker and erased at run time (a `T` is checked as `any`, a `Box(int)` as a `Box`).
The JIT compiles a `forall` function once per type it is called with (`max` for
`int`, `max` for `float`), as fast as a function written for that type; likewise the
methods of a generic class, per type of its fields.

```ruby
def first(xs: [T]): T? forall T = xs[0]?

def map(xs: [T], f: (T): U): [U] forall T, U
  let out: [U] = []
  for x in xs
    out << f(x)
  end
  out
end

first([1, 2])                      # int?
map(["a", "bcd"], |s| s.len())     # [int]
```

- `forall T, U` after the signature declares the parameters of a function or a
  method. At each call they are found from the arguments (then from the expected
  type: `let xs: [int] = empty()`).
- In the body, a `T` is any type, the same everywhere: `def f(x: T): T = 1` is an
  error.

### Generic types

```ruby
class Box(T)
  def init(@value: T)
  end
  def get(): T = @value
  def set(v: T) = @value = v
  def map(f: (T): U): Box(U) forall U = Box(f(@value))
end

type Pair(A, B)
  a: A
  b: B
end

let b: Box(int) = Box(1)           # `T` found from `1`
b.get() + 1                        # int
b.set("x")                         # error: expected type `int`
let p = Pair(a: 1, b: "z")         # Pair(int, str)

class IntBox < Box(int)            # a generic parent, with its arguments
end
class Labeled(T) < Box(T)
end
```

- A `class`, a `type` or a `trait` takes parameters after its name: `Box(T)`, with no
  space. `Box(int)` in an annotation gives the arguments; `Box` alone is `Box(any)`.
- At construction, the arguments are found from the values, then from the expected
  type (`let s: Stack(int) = Stack()`).
- A generic type is compatible with another one if its arguments are
  (`Box(int)` is not a `Box(str)`; an `IntBox` is a `Box(int)`).

### Generic traits and bounds

```ruby
trait Comparable(T)
  abstract def cmp(o: T): int
  def lt(o: T): bool = self.cmp(o) < 0
end

class Money
  include Comparable(Money)
  def init(@cents: int)
  end
  def cents = @cents
  def cmp(o: Money): int = @cents - o.cents
end

def max(a: T, b: T): T forall T: Comparable(T)
  if a.cmp(b) >= 0 then a else b end
end

max(Money(3), Money(5))           # Money
max(1, 2)                         # error: `int` (for `T`) is not a `Comparable(int)`
```

- `T: Bound` (in `forall` or after a type's name: `class Sorted(T: Comparable(T))`)
  restricts `T`: in the body, a `T` has the methods of its bound; at a call or in an
  annotation, the type found must satisfy it.
- Without a bound, a `T` can do anything in the body, checked at run time (gradual
  typing).

## Modules

A `module` groups declarations under a name: constants, functions, types, classes,
traits, enums, other modules.

```ruby
module Http
  let VERSION = "1.1"

  class Request
    def init(@url: str)
    end
    def url = @url
  end

  class Failure < Error
  end

  def get(url: str): Request = Request(url)   # a function of the module

  module Mime
    let JSON = "application/json"
  end
end

let r: Http::Request = Http.get("example.org")   # `::` for a member, `.` for a call
Http::VERSION                                    # "1.1"
Http::Mime::JSON
class ApiError < Http::Failure                    # also after `<` and `include`
end
```

- Outside, a member is named `Module::name`; a function of the module is called with
  `Module.f(…)`. A module is not a value (`let m = Http` is an error).
- Inside, short names are looked up in the module first, then outside: the module's
  functions can call the script's functions.
- A module's body runs when the block that contains it starts, before the block's
  statements (like the declarations of types): its statements cannot use the script's
  variables, its functions can (they run later).
- A module can be reopened, also by a later script, to add members.
- `private` before a declaration keeps it inside the module (and the modules nested
  in it): `Http.sign(…)` or `Http::Conn` outside is an error, reported before running.

```ruby
module Http
  private let TOKEN = "t0k"
  private def sign(s: str): str = "#{s}+#{TOKEN}"
  def get(url: str): str = sign(url)
end

Http.get("a.org")      # "a.org+t0k"
Http.sign("a.org")     # error: `sign` is private to module `Http`
```
- A module is declared at the top level of the script, or in a module.

### Files: `require`

```ruby
require "lib/http"            # lib/http.bk, relative to this file

Http.get("example.org")
```

- A required file is loaded once (by its name), before the script is checked: its
  declarations are known as those of a previous script. Nested `require`s are loaded
  first; a cycle is an error.
- `require` is written at the top level of the script, with a literal name.
- Embedded, the files are provided by the host (`Engine::set_loader`); the `bokoni`
  command reads them from disk (`.bk` added when the name has no extension).

## Comments

```ruby
# single-line comment
```

## Annotations

> Inspired by [Crystal](https://crystal-lang.org/reference/1.22/syntax_and_semantics/annotations/index.html).

An annotation attaches metadata to a declaration. By itself it changes nothing at run time: it is read by the host Rust code (and, later, by macros).

### Declaration

An annotation is declared like a `type`: its arguments are therefore typed and checked.

```ruby
annotation Validate
  min: int? = nil
  max: int? = nil
end

annotation Column
  name: str? = nil
  primary: bool = false
end

annotation Tag                # no argument
end
```

### Application

`@[Name]` or `@[Name(args)]`, placed before the declaration. No conflict with `@field`: `@[` is a distinct token.

```ruby
@[Tag]
type Server
  @[Validate(min: 1, max: 65535)]
  port: int = 80

  @[Column(name: "host_name")]
  @[Deprecated("use address")]
  host: str = "localhost"
end

@[Deprecated("use add")]
def plus(a: int, b: int): int = a + b

def connect(@[Validate(min: 0)] timeout: int)
end

enum Protocol: str
  @[Deprecated]
  Ftp = "FTP"
  Tcp = "TCP"
end
```

- Targets: `type`, `class`, `enum`, enum variant, field, method, function, parameter.
- Named arguments (`min: 1`) or positional ones in field order (`Validate(1, 65535)`).
- Several annotations, including of the same type, on the same target.
- Arguments are checked before running, like a construction: unknown annotation,
  nonexistent field, argument type, missing required field.
- They are evaluated **when read** (from Rust), not at declaration; they are constants:
  they may use globals (`@[Limit(MAX + 2)]`), not local variables.

### Reading

In a macro (planned: version 3):

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

- `x.annotation(A)`: the last `A` annotation (or `nil`).
- `x.annotations(A)` / `x.annotations()`: all `A` annotations / all of them.
- Arguments are read like fields: `v.min`, `v.max`.

From Rust:

```rust
let ty = engine.type_info("Server")?;          // also: function_info, enum_info
for field in ty.fields() {
    if let Some(v) = field.annotation("Validate") {
        let min: Option<i64> = v.get("min")?;
    }
}
let doc = ty.doc();                              // text of @[Doc("…")]
let deprecated = ty.field("host").unwrap().deprecated(); // Some(Some("use address"))
```

`type_info`, `function_info` and `enum_info` return a description (fields with their type,
methods, parameters, variants with their value) that no longer depends on the engine. On
each element: `annotation(name)` (the last one), `annotations(name)`, `all_annotations()`,
`doc()`, `deprecated()`; on an annotation: `name()`, `fields()`, `get::<T>(field)`.

### Built-in annotations

| Annotation             | Effect                                                  |
|------------------------|---------------------------------------------------------|
| `@[Deprecated(msg?)]`  | warning on every use                                    |
| `@[Doc("…")]`          | documentation (alternative to comments)                 |

They are declared by the engine as ordinary annotations
(`Deprecated(message: str? = nil)`, `Doc(text: str)`).

The `Deprecated` warning is emitted by the checker, before running, on every use: call or
mention of a function, a type, an enum, a variant (`Protocol::Ftp`), read of a field or
passing it by name, method call, argument passed to a deprecated parameter — including
from a later script. From Rust: `engine.take_warnings()` or
`engine.set_warning_handler(…)`; the command line prints them on standard error.

```text
warning: `plus` is deprecated: use add
 --> script.bk:17:7
   |
17 | print(plus(1, 2))
   |       ^^^^
```

## Macros (planned: version 3)

> Inspired by [Crystal](https://crystal-lang.org/reference/1.22/syntax_and_semantics/macros/index.html).

A macro runs **at compile time**: it receives AST nodes (not values) and generates code that is inserted in its place.

- `{{ expr }}`: inserts a node / a value into the generated code.
- `{% … %}`: logic run at compile time (`if`, `for`, `let`).

```ruby
macro getters(*names)
  {% for name in names %}
    def {{name}} = @{{name}}
  {% end %}
end

class User
  def init(@name: str, @email: str)
  end

  getters(name, email)        # generates: def name = @name / def email = @email
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

### Context available in a macro

| Variable     | Content                                               |
|--------------|-------------------------------------------------------|
| `@type`      | the current type: `name`, `fields`, `methods`, `annotation(…)` |
| `@def`       | the current method: `name`, `params`, `return_type`, `annotation(…)` |
| `name.id`    | a node converted to an identifier                     |
| `node.stringify` | the source code of a node, as a `str`             |

### Rules

- The generated code must be valid on its own (no lone `when` outside a `case`).
- Hygiene: variables declared with `let` in the generated code do not leak into the caller (automatic renaming).
- Errors point both to the macro call and to the faulty generated line.
- A macro is visible in the file / type where it is defined, and in its descendants.

### Prerequisites from version 1

- Complete AST with source positions (file, line, column) on every node.
- AST that can be serialized and printed back as source code (`stringify`).
- The parser accepts and keeps `@[…]` (done: annotations are read from Rust).

## Reserved words

`let def end if elsif else unless then case when while for in do break next return
module require private protected forall
begin rescue ensure raise super trait abstract include type class enum annotation macro true false nil and or not`

## Open questions

- Variance of generic types: compatible when their arguments are (covariant, like
  lists), which is lenient for a `Box(T)` that can be modified.
- Type arguments kept at run time (`Acc(int)`) so that typed native code can call the
  methods of a generic object directly.
