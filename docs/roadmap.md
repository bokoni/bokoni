# Feuille de route

Objectif final : un langage de script intégré à Rust, avec un JIT (Cranelift) pour le code typé.

## Architecture visée

```
source ─► lexer ─► parseur ─► AST ─► vérificateur de types ─► IR typée ─┬─► interpréteur
                                                                         └─► JIT (Cranelift)
```

- **IR typée** : représentation intermédiaire simple où chaque valeur a un type connu.
  Une fonction entièrement typée (`int`, `float`, `bool`) est compilée en code natif ;
  le code `any` manipule des valeurs génériques (« boxées ») vérifiées à l'exécution.
- **L'interpréteur reste** même avec le JIT : exécution du code `any`, plateformes sans JIT
  (WebAssembly, iOS), et référence pour tester le JIT (mêmes résultats).

## Paliers

### v0.1 — Cœur exécutable

- [x] Lexer écrit à la main : octets, à la demande (un jeton d'avance), interpolation `"#{…}"`
- [x] Parseur écrit à la main (descente récursive + Pratt pour les opérateurs)
- [x] AST compact : expressions dans un `Vec`, désignées par index (`ExprId`), 32 octets
      chacune, positions source sur chaque nœud, affichable (S-expressions)
- [x] Pas de débordement de pile possible (parseur et interpréteur : `stacker`)
- [x] Le parseur accepte toute la syntaxe de `syntax.md` (y compris `type`, `class`, `enum`, `@[…]`)
- [x] Messages d'erreur avec ligne, colonne et extrait du source
- [x] Interpréteur sur l'AST : `let`, `def`, `if`, `case`, `while`, `for`, lambdas, blocs `do`
- [x] Valeurs : `int`, `float`, `str`, `bool`, `nil`, listes, hash, records, tuples, intervalles
- [x] Annotations de type vérifiées à l'exécution (`let`, paramètres, retour)
- [x] API `Engine` pour Rust : `eval`, `register_fn` avec conversion automatique des types
- [x] Exécutable : exécuter un fichier, REPL, `--ast` pour afficher l'arbre
- [x] Récupération d'erreurs dans le parseur (toutes les erreurs en une passe) — fait en v0.2

### v0.2 — Typage statique

- [x] Toutes les erreurs signalées en une passe : le lexer et le parseur reprennent après
      une erreur (saut jusqu'au `end` des constructions ouvertes, sans cascade)
- [x] Vérification des noms (portées, `let`, constantes, utilisation avant déclaration,
      champs des blocs de construction), avec suggestions (`len` pour `lenght`)
- [x] Vérificateur de types : inférence locale, `any`, `T?`, unions, types littéraux ;
      seules les erreurs certaines sont signalées (typage graduel)
- [x] Les déclarations des scripts précédents (REPL) sont connues du vérificateur
- [x] `type`, `enum`, `class` à l'exécution ; blocs `Server do … end`
- [x] Conversion record littéral → type nominal
- [ ] Types génériques (`[T]` dans les signatures), héritage / interfaces, modules

### v0.3 — IR typée

- [ ] Traduction AST vérifié → IR (fonctions, blocs de base, instructions typées)
- [ ] L'interpréteur exécute l'IR (plus rapide, même format que le JIT)

### v0.4 — JIT Cranelift

- [ ] Compilation native des fonctions entièrement typées, interprétation des autres
- [ ] Appels JIT ↔ interpréteur ↔ fonctions Rust avec une convention unique
- [ ] Compilation à chaud (fonctions souvent appelées)
- Référence : `piss/createJit/resources/examples/cranelift-jit-demo`

### Ensuite

- Annotations lisibles depuis Rust (v2 de `syntax.md`)
- Macros à la Crystal (v3 de `syntax.md`)

## Décisions qui engagent le JIT

| Sujet                    | Choix v0.1                                          |
|--------------------------|-----------------------------------------------------|
| Représentation des valeurs | `enum Value` (24 octets), types de base sans allocation ; NaN-boxing (8 octets) à étudier pour le JIT |
| Mémoire                  | comptage de références (`Rc`), pas de ramasse-miettes |
| Fonctions Rust           | convention unique `Fn(&[Value]) -> Result<Value, String>` |
| Noms                     | interning : `Symbol(u32)`, comparaisons en O(1)      |
| Positions                | `Span { start: u32, end: u32 }` en octets            |
| AST                      | compact, nœuds désignés par index : une closure ne garde que `Rc<Script>` + index |

## Performances du parseur

`cargo run --release --example bench_parser` compare le parseur à une grammaire
[`peg`](https://crates.io/crates/peg) équivalente (sous-ensemble commun : `let`, assignation,
opérateurs, appels, champs, index, listes, chaînes). La version peg fait *moins* de travail
(pas de positions, pas d'interning).

| Mesure                                  | bokoni        | peg          |
|-----------------------------------------|---------------|--------------|
| Fichier de 150 000 lignes (9 Mo)        | ~60 ms (150 Mo/s) | ~93 ms (100 Mo/s) |
| Petit script (60 lignes) × 20 000       | ~400 ms       | ~820 ms      |
| Allocations par ligne                   | 0,34          | 14,7         |

Ce qui comptait, mesuré au profileur : la quantité de mémoire touchée. Passer d'une `Box`
par nœud à l'AST compact a divisé le temps de parsing par deux (120 ms → 60 ms) ; les autres
optimisations (lexer à la demande, interning des noms courts) n'ont apporté que quelques %.

## Limites connues (v0.1)

- Cycles de références : une fonction locale qui se capture elle-même n'est jamais libérée
  (les fonctions globales le sont, à la destruction de l'`Engine`).
- L'interpréteur parcourt l'AST et copie les listes avant un `for` : correct, pas rapide.
  La vitesse d'exécution est l'objet de la v0.3 (IR) et de la v0.4 (JIT).
