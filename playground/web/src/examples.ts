// Example programs offered in the playground (each one runs without errors).

export interface Example {
  name: string;
  code: string;
}

export const examples: Example[] = [
  {
    name: "Hello, Bokoni",
    code: `# Bokoni: Ruby-like syntax, gradual typing.
# Press Ctrl+Enter (⌘+Enter on a Mac) to run.

def greet(who: str): str = "Hello, #{who}!"

print(greet("world"))

let squares = [1, 2, 3, 4, 5].map(|n| n * n)
print("squares: #{squares}")
print("sum: #{squares.sum()}")

# The last expression is the program's value.
squares.filter(|n| n % 2 == 1)
`,
  },
  {
    name: "Types and classes",
    code: `# \`type\`: data with named fields and default values.
type Point
  x: float = 0.0
  y: float = 0.0
end

# \`class\`: objects with an \`init\` and methods.
class Account
  def init(@owner: str, @balance: int = 0)
  end

  def deposit(amount: int): int
    raise "amount must be positive" if amount <= 0
    @balance += amount
    @balance
  end

  def to_s(): str = "#{@owner}: #{@balance}"
end

let p = Point(x: 1.5)
let a = Account("Ada")
a.deposit(100)
a.deposit(42)
print(a.to_s())
[p.x, p.y, a.deposit(8)]
`,
  },
  {
    name: "Inheritance and traits",
    code: `trait Shape
  abstract def area(): float
  def describe(): str = "#{typeof(self)} of area #{self.area()}"
end

class Circle
  include Shape
  def init(@r: float)
  end
  def area(): float = 3.14159 * @r * @r
end

class Square
  include Shape
  def init(@side: float)
  end
  def area(): float = @side * @side
end

let shapes: [Shape] = [Circle(1.0), Square(2.0)]
for s in shapes
  print(s.describe())
end
shapes.map(|s| s.area()).sum()
`,
  },
  {
    name: "Enums and case",
    code: `enum Color: str
  Red = "red"
  Green = "green"
  Blue = "blue"
end

def warm?(c: Color): bool
  case c
  when Color::Red then true
  else false
  end
end

# \`case\` by type: the subject is refined in each arm.
def label(x: int | str | nil): str
  case x
  when int then "the number #{x}, doubled: #{x * 2}"
  when str then "the text \\"#{x}\\" (#{x.len()} chars)"
  else "nothing"
  end
end

print(Color.values().map(|c| c.value))
print(label(42))
print(label("hi"))
print(label(nil))
[warm?(Color::Red), warm?(Color::Blue)]
`,
  },
  {
    name: "Nil safety",
    code: `type User
  name: str
  email: str? = nil
end

def domain(u: User): str?
  # \`u.email\` may be nil: the checker refuses \`.split\` on it
  # until it is tested.
  return nil if u.email.nil?
  u.email.split("@")[1]
end

let users = [User(name: "ada", email: "ada@example.org"), User(name: "bob")]
for u in users
  print("#{u.name}: #{domain(u) ?? "no email"}")
end

# Safe navigation: nil as soon as a link is nil.
users.map(|u| u.email?.len())
`,
  },
  {
    name: "Errors",
    code: `class NotFound < Error
end

def find(xs: [int], x: int): int
  for i in 0...xs.len()
    return i if xs[i] == x
  end
  raise NotFound("#{x} is not in the list")
end

let xs = [4, 8, 15, 16, 23, 42]
let results = [15, 99].map(|x| begin
  "found at #{find(xs, x)}"
rescue e: NotFound
  "error: #{e.message}"
end)
print(results)

# Remove the comment below to see a type error, caught before running:
# find(xs, "15")
results.len()
`,
  },
  {
    name: "Generics",
    code: `class Stack(T)
  def init(@items: [T] = [])
  end
  def push(x: T): Stack(T)
    @items << x
    self
  end
  def pop(): T? = @items.pop()
  def size(): int = @items.len()
end

def first(xs: [T]): T? forall T = xs[0]?

let s: Stack(str) = Stack([])
s.push("a").push("b").push("c")
print("top: #{s.pop()}, size: #{s.size()}")
[first([3, 1, 2]), first([])]
`,
  },
  {
    name: "Speed test",
    code: `# The interpreter only, here (native compilation is not available
# in WebAssembly): typed or not, the same code.
def fib(n: int): int
  if n < 2 then n else fib(n - 1) + fib(n - 2) end
end

def primes(n: int): [int]
  let out: [int] = []
  for i in 2..n
    let prime = true
    for p in out
      if p * p > i then break end
      if i % p == 0
        prime = false
        break
      end
    end
    out << i if prime
  end
  out
end

print("fib(22) = #{fib(22)}")
let ps = primes(20000)
print("#{ps.len()} primes up to 20000, the last: #{ps[-1]}")
`,
  },
];
