**Star Wars Named Entity Recognition (NER) Annotation Guidelines** 🚀⭐️🪐

**1. Introduction** 🚀⭐️🪐
These guidelines define how to annotate entities in Star Wars–related text using the BI (Begin–Inside) tagging scheme with 4 entity types: PER (person), ORG (organization), LOC (location), and MISC (miscellaneous). Consistent, high-quality annotations help train robust NER models specific to the Star Wars universe. 🚀⭐️🪐

**2. Tagging Scheme** 🚀⭐️🪐

- **B-**: Marks the first token of an entity of type TYPE.
- **I-**: Marks each subsequent token within the same entity.
- **O**: Marks tokens that are not part of any entity.

**3. Entity Types & Definitions** 🚀⭐️🪐

- **PER (Person)**: Named characters (e.g., Luke Skywalker, Darth Vader, Ahsoka Tano).
- **ORG (Organization)**: Formal groups, bodies, or alliances (e.g., Galactic Empire, Rebel Alliance, Jedi Order).
- **LOC (Location)**: Planets, star systems, cities, regions (e.g., Tatooine, Coruscant, Dagobah).
- **MISC (Miscellaneous)**: Other named entities of significance, including species (Wookiee), vehicles (Millennium Falcon), weapons (Lightsaber), events (Battle of Endor), droids (R2-D2).

**4. Annotation Process** 🚀⭐️🪐

1. **Tokenization**: Split text on whitespace and punctuation, keeping punctuation tokens separate.
2. **Span Identification**: For each entity span, assign B- and I- tags across all tokens.
3. **Non-Overlapping**: Entities cannot overlap or nest. Choose the longest valid span if ambiguity arises.
4. **Consistency**: Use the same label whenever the same entity appears in different contexts.

**5. Detailed Examples** 🚀⭐️🪐

| Sentence                                                          | Tokens                                                                                      | Tags                                                                     |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Luke Skywalker flies the Millennium Falcon to Tatooine.           | Luke / Skywalker / flies / the / Millennium / Falcon / to / Tatooine / .                    | B-PER / I-PER / O / O / B-MISC / I-MISC / O / B-LOC / O                  |
| The Rebel Alliance’s victory at the Battle of Endor was decisive. | The / Rebel / Alliance / ’s / victory / at / the / Battle / of / Endor / was / decisive / . | O / B-ORG / I-ORG / O / O / O / O / B-MISC / I-MISC / I-MISC / O / O / O |

**6. Edge Cases & Best Practices** 🚀⭐️🪐

- **Hyphenated Names**: Treat each part separately with B- and I- (e.g., B-PER Grand / I-PER Moff).
- **Possessives**: If an entity is in possessive form (Rebel Alliance’s), annotate only the entity tokens: Rebel (B-ORG), Alliance (I-ORG); ignore ’s.
- **Abbreviations & Acronyms**: Annotate as ORG or MISC based on type (e.g., B-ORG ISB for Imperial Security Bureau).
- **Titles & Honorifics**: Include titles if they are commonly part of the name (e.g., B-PER Chancellor / I-PER Palpatine).
- **Nested Phrases**: Do not nest. In “Jedi Council Temple,” tag the entire phrase as LOC rather than ORG+LOC.
- **Ambiguity**: When unsure, default to MISC and flag for review.

**7. Multi-Word Entities** 🚀⭐️🪐

- All contiguous tokens forming a single entity must be tagged with B- for the first token and I- for the rest.
- Separate entities by at least one O-labeled token.

**8. Quality Control & Review** 🚀⭐️🪐

- **Double Annotation**: Have two annotators label the same text and resolve conflicts.
- **Spot Checks**: Randomly sample annotated sentences for consistency.

**9. Annotation Tool Tips** 🚀⭐️🪐

- Use a tool that supports start–end offsets and BI tags.
- Leverage pre-tokenization to speed up labeling.

**10. Glossary of Common Star Wars Entities** 🚀⭐️🪐

- **PER**: Luke Skywalker, Leia Organa, Han Solo, Obi-Wan Kenobi, Yoda, Boba Fett
- **ORG**: Galactic Republic, Separatists, First Order, Sith Order, Mandalorians
- **LOC**: Hoth, Naboo, Endor, Mustafar, Bespin
- **MISC**: TIE Fighter, Death Star, Force, Mind Trick, Clone Trooper
