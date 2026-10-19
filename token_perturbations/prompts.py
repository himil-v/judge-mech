"""
GPT-4o prompt templates for the eight-attack perturbation taxonomy (Section 3, Table 1).

Each template contains three category-specific few-shot exemplars (intensity 1, 2, 3)
followed by the target summary. Placeholders filled at runtime:
    {right_summary}  the clean summary to perturb
    {intensity}      k, the number of errors to inject (all paper experiments use k = 1)

The template text is kept verbatim as used to generate the released corpus.

Attack keys (used in data filenames) and their names in the paper:
    read_preposition    Preposition Mismatch    (Readability)
    read_tense          Tense Mismatch          (Readability)
    read_spelling       Spelling                (Readability)
    read_reordering     Sequential Reordering   (Readability)
    adeq_entity_1to1    Entity Swap             (Adequacy)
    adeq_numerical      Numerical/Date Swap     (Adequacy)
    adeq_coreference    Coreference Mismatch    (Adequacy)
    adeq_polarity_swap  Antonym & Negation      (Adequacy; "Polarity" in Tables 2-4)
"""

prompt_read_preposition = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Preposition Mismatch" grammatical error(s) into the provided Original text.

Rule 1: Find {intensity} correct preposition(s) in the text and swap them for incorrect ones that violate English collocation rules (e.g., swapping "on" for "at", or "in" for "on").
Rule 2: You must NEVER alter, add, or delete any other words or punctuation in the sentence. The word count and structural index of all other words must remain absolutely identical.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
The United Nations Security Council has voted on the new ceasefire resolution in New York.
Intensity:
1
Output:
{{
  "original_summary": "The United Nations Security Council has voted on the new ceasefire resolution in New York.",
  "perturbed_summary": "The United Nations Security Council has voted at the new ceasefire resolution in New York.",
  "original_tokens": ["on"],
  "new_tokens": ["at"],
  "perturbation_type": "Readability_Preposition_Mismatch"
}}

Original text:
A massive wildfire has broken out in the central region of California.
Intensity:
2
Output:
{{
  "original_summary": "A massive wildfire has broken out in the central region of California.",
  "perturbed_summary": "A massive wildfire has broken out on the central region to California.",
  "original_tokens": ["in", "of"],
  "new_tokens": ["on", "to"],
  "perturbation_type": "Readability_Preposition_Mismatch"
}}

Original text:
The tech company announced a new flagship smartphone during the annual developer conference at their headquarters.
Intensity:
3
Output:
{{
  "original_summary": "The tech company announced a new flagship smartphone during the annual developer conference at their headquarters.",
  "perturbed_summary": "The tech company announced a new flagship smartphone on the annual developer conference in their headquarters.",
  "original_tokens": ["during", "at"],
  "new_tokens": ["on", "in"],
  "perturbation_type": "Readability_Preposition_Mismatch"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

prompt_read_tense = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Tense Contradiction" grammatical error(s) into the provided Original text.

Rule 1: Find {intensity} verb(s) in the text and change their tense to create a jarring grammatical mismatch within the sentence context (e.g., changing a past tense verb to present tense in a sentence describing a past event, or mixing tenses inappropriately).
Rule 2: You must NEVER alter, add, or delete any other words. The total word count of the sentence must remain absolutely identical. Only swap the target verb(s) for a different conjugated form of the exact same verb.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
The prime minister visited the hospital yesterday and spoke to the nurses.
Intensity:
1
Output:
{{
  "original_summary": "The prime minister visited the hospital yesterday and spoke to the nurses.",
  "perturbed_summary": "The prime minister visits the hospital yesterday and spoke to the nurses.",
  "original_tokens": ["visited"],
  "new_tokens": ["visits"],
  "perturbation_type": "Readability_Tense_Contradiction"
}}

Original text:
The team secured their victory after the striker scored a goal in the final minute.
Intensity:
2
Output:
{{
  "original_summary": "The team secured their victory after the striker scored a goal in the final minute.",
  "perturbed_summary": "The team secures their victory after the striker scoring a goal in the final minute.",
  "original_tokens": ["secured", "scored"],
  "new_tokens": ["secures", "scoring"],
  "perturbation_type": "Readability_Tense_Contradiction"
}}

Original text:
The scientists discovered a new species of frog that lives in the rainforest and eats insects.
Intensity:
3
Output:
{{
  "original_summary": "The scientists discovered a new species of frog that lives in the rainforest and eats insects.",
  "perturbed_summary": "The scientists discover a new species of frog that lived in the rainforest and eat insects.",
  "original_tokens": ["discovered", "lives", "eats"],
  "new_tokens": ["discover", "lived", "eat"],
  "perturbation_type": "Readability_Tense_Contradiction"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

prompt_adeq_entity_1to1 = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Entity Hallucination (1-to-1 Token)" error(s) into the provided Original text.

Rule 1: Find {intensity} single-word named entity/entities (Person, Organization, or Location) in the text and replace them with different single-word entities of the exact same category (e.g., swap "Apple" for "Google", or "London" for "Paris").
Rule 2: The swapped entity MUST be a single word to preserve a 1-to-1 word swap. You must NEVER alter, add, or delete any other words or punctuation. The total word count of the sentence must remain absolutely identical.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
Tim Cook announced a new flagship store opening in California today.
Intensity:
1
Output:
{{
  "original_summary": "Tim Cook announced a new flagship store opening in California today.",
  "perturbed_summary": "Tim Cook announced a new flagship store opening in Texas today.",
  "original_tokens": ["California"],
  "new_tokens": ["Texas"],
  "perturbation_type": "Adequacy_Entity_1to1"
}}

Original text:
The BBC has reported that Microsoft is aggressively acquiring the new startup.
Intensity:
2
Output:
{{
  "original_summary": "The BBC has reported that Microsoft is aggressively acquiring the new startup.",
  "perturbed_summary": "The CNN has reported that Google is aggressively acquiring the new startup.",
  "original_tokens": ["BBC", "Microsoft"],
  "new_tokens": ["CNN", "Google"],
  "perturbation_type": "Adequacy_Entity_1to1"
}}

Original text:
Obama met with the prime minister of France during his visit to Paris.
Intensity:
3
Output:
{{
  "original_summary": "Obama met with the prime minister of France during his visit to Paris.",
  "perturbed_summary": "Biden met with the prime minister of Spain during his visit to Berlin.",
  "original_tokens": ["Obama", "France", "Paris"],
  "new_tokens": ["Biden", "Spain", "Berlin"],
  "perturbation_type": "Adequacy_Entity_1to1"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

prompt_adeq_antonym_swap = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Polarity/Antonym Swap" error(s) into the provided Original text.

Rule 1: Find {intensity} word(s) (such as verbs, adjectives, or adverbs) that convey a clear semantic direction, action, or sentiment, and swap them with their direct antonyms to completely reverse the meaning of the clause (e.g., changing "increased" to "decreased", "won" to "lost", or "best" to "worst").
Rule 2: The swapped word(s) MUST exactly match the grammatical form of the original word(s) so the sentence remains grammatically perfect. You must NEVER alter, add, or delete any other words or punctuation. The total word count of the sentence must remain absolutely identical.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
The company announced that their quarterly profits have significantly increased this year.
Intensity:
1
Output:
{{
  "original_summary": "The company announced that their quarterly profits have significantly increased this year.",
  "perturbed_summary": "The company announced that their quarterly profits have significantly decreased this year.",
  "original_tokens": ["increased"],
  "new_tokens": ["decreased"],
  "perturbation_type": "Adequacy_Polarity_Swap"
}}

Original text:
The home team easily won the championship game against their bitter rivals.
Intensity:
2
Output:
{{
  "original_summary": "The home team easily won the championship game against their bitter rivals.",
  "perturbed_summary": "The home team easily lost the championship game against their friendly rivals.",
  "original_tokens": ["won", "bitter"],
  "new_tokens": ["lost", "friendly"],
  "perturbation_type": "Adequacy_Polarity_Swap"
}}

Original text:
The new policy is expected to create a much faster and more reliable system.
Intensity:
3
Output:
{{
  "original_summary": "The new policy is expected to create a much faster and more reliable system.",
  "perturbed_summary": "The old policy is expected to create a much slower and more unreliable system.",
  "original_tokens": ["new", "faster", "reliable"],
  "new_tokens": ["old", "slower", "unreliable"],
  "perturbation_type": "Adequacy_Polarity_Swap"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

prompt_read_spelling = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Spelling Mistake" error(s) into the provided Original text.

Rule 1: Find {intensity} word(s) in the text and introduce a realistic typographical or spelling error (e.g., dropping a letter, swapping adjacent letters, or adding a common typo) while keeping the word mostly recognizable.
Rule 2: You must NEVER alter, add, or delete any other words or punctuation in the sentence. The total word count and structural index of all words must remain absolutely identical.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
The government announced a new environmental policy today.
Intensity:
1
Output:
{{
  "original_summary": "The government announced a new environmental policy today.",
  "perturbed_summary": "The goverment announced a new environmental policy today.",
  "original_tokens": ["government"],
  "new_tokens": ["goverment"],
  "perturbation_type": "Readability_Spelling_Mistake"
}}

Original text:
The scientists discovered a major breakthrough in medical research.
Intensity:
2
Output:
{{
  "original_summary": "The scientists discovered a major breakthrough in medical research.",
  "perturbed_summary": "The scienists discovered a major breakthough in medical research.",
  "original_tokens": ["scientists", "breakthrough"],
  "new_tokens": ["scienists", "breakthough"],
  "perturbation_type": "Readability_Spelling_Mistake"
}}

Original text:
A beautiful environment is essential for building a healthy community.
Intensity:
3
Output:
{{
  "original_summary": "A beautiful environment is essential for building a healthy community.",
  "perturbed_summary": "A beatiful enviroment is esential for building a healthy community.",
  "original_tokens": ["beautiful", "environment", "essential"],
  "new_tokens": ["beatiful", "enviroment", "esential"],
  "perturbation_type": "Readability_Spelling_Mistake"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

prompt_adeq_numerical = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Numerical/Date Swap" error(s) into the provided Original text.

Rule 1: Find {intensity} single-word numerical value(s), date(s), day(s), or quantities (e.g., "50", "million", "2023", "Tuesday") and replace them with different values of the exact same category.
Rule 2: The swapped value MUST be a single word to preserve a 1-to-1 word swap. You must NEVER alter, add, or delete any other words or punctuation. The total word count of the sentence must remain absolutely identical.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
The tech giant reported a net profit of 50 million dollars.
Intensity:
1
Output:
{{
  "original_summary": "The tech giant reported a net profit of 50 million dollars.",
  "perturbed_summary": "The tech giant reported a net profit of 40 million dollars.",
  "original_tokens": ["50"],
  "new_tokens": ["40"],
  "perturbation_type": "Adequacy_Numerical_Date_Swap"
}}

Original text:
On Tuesday, the global stock market dropped by 5 percent.
Intensity:
2
Output:
{{
  "original_summary": "On Tuesday, the global stock market dropped by 5 percent.",
  "perturbed_summary": "On Friday, the global stock market dropped by 9 percent.",
  "original_tokens": ["Tuesday", "5"],
  "new_tokens": ["Friday", "9"],
  "perturbation_type": "Adequacy_Numerical_Date_Swap"
}}

Original text:
In 2022, the regional population grew by 3 million people across 10 cities.
Intensity:
3
Output:
{{
  "original_summary": "In 2022, the regional population grew by 3 million people across 10 cities.",
  "perturbed_summary": "In 2024, the regional population grew by 7 million people across 12 cities.",
  "original_tokens": ["2022", "3", "10"],
  "new_tokens": ["2024", "7", "12"],
  "perturbation_type": "Adequacy_Numerical_Date_Swap"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

prompt_adeq_coreference = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Coreference Mismatch Hallucination" error(s) into the provided Original text.

Rule 1: Find {intensity} pronoun(s) or possessive determiner(s) (e.g., "he", "she", "his", "their", "it") and swap them with incorrect pronouns to break the reference chain or create a gender/plurality mismatch.
Rule 2: The swapped pronoun MUST be a single word to preserve a 1-to-1 word swap. You must NEVER alter, add, or delete any other words or punctuation. The total word count of the sentence must remain absolutely identical.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
John lost his wallet at the train station yesterday.
Intensity:
1
Output:
{{
  "original_summary": "John lost his wallet at the train station yesterday.",
  "perturbed_summary": "John lost her wallet at the train station yesterday.",
  "original_tokens": ["his"],
  "new_tokens": ["her"],
  "perturbation_type": "Adequacy_Coreference_Mismatch"
}}

Original text:
The queen waved to the massive crowd as she entered her royal carriage.
Intensity:
2
Output:
{{
  "original_summary": "The queen waved to the massive crowd as she entered her royal carriage.",
  "perturbed_summary": "The queen waved to the massive crowd as he entered his royal carriage.",
  "original_tokens": ["she", "her"],
  "new_tokens": ["he", "his"],
  "perturbation_type": "Adequacy_Coreference_Mismatch"
}}

Original text:
When the brothers found the stray dog, they immediately gave it their food.
Intensity:
3
Output:
{{
  "original_summary": "When the brothers found the stray dog, they immediately gave it their food.",
  "perturbed_summary": "When the brothers found the stray dog, we immediately gave him our food.",
  "original_tokens": ["they", "it", "their"],
  "new_tokens": ["we", "him", "our"],
  "perturbation_type": "Adequacy_Coreference_Mismatch"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

prompt_read_reordering = """You are a highly precise linguistic perturbation engine.
Your task is to inject exactly {intensity} "Sequential Word Reordering" grammatical error(s) into the provided Original text.

Rule 1: Find {intensity} pair(s) of adjacent words in the text and swap their positions to create a noticeable grammatical or syntactic disruption (e.g., swapping a noun and its adjective like "brown dog" to "dog brown", or a verb and its article like "visited the" to "the visited").
Rule 2: You must NEVER alter, add, or delete any words or punctuation. The vocabulary and total word count of the sentence must remain absolutely identical; only the structural sequence of the targeted adjacent words should change.
Rule 3: You must output your response in strictly valid JSON format. Do not include markdown formatting like ```json or ``` in your final output, just the raw JSON object.

Original text:
The government announced a new environmental policy today.
Intensity:
1
Output:
{{
  "original_summary": "The government announced a new environmental policy today.",
  "perturbed_summary": "The government announced new a environmental policy today.",
  "original_tokens": ["a", "new"],
  "new_tokens": ["new", "a"],
  "perturbation_type": "Readability_Sequential_Reordering"
}}

Original text:
The prime minister visited the hospital yesterday and spoke to the nurses.
Intensity:
2
Output:
{{
  "original_summary": "The prime minister visited the hospital yesterday and spoke to the nurses.",
  "perturbed_summary": "The prime minister the visited hospital yesterday and spoke the to nurses.",
  "original_tokens": ["visited", "the", "to", "the"],
  "new_tokens": ["the", "visited", "the", "to"],
  "perturbation_type": "Readability_Sequential_Reordering"
}}

Original text:
A massive wildfire has broken out in the central region of California.
Intensity:
3
Output:
{{
  "original_summary": "A massive wildfire has broken out in the central region of California.",
  "perturbed_summary": "A wildfire massive has out broken in the central of region California.",
  "original_tokens": ["massive", "wildfire", "broken", "out", "region", "of"],
  "new_tokens": ["wildfire", "massive", "out", "broken", "of", "region"],
  "perturbation_type": "Readability_Sequential_Reordering"
}}

Original text:
{right_summary}
Intensity:
{intensity}
Output:
"""

# Attack key -> prompt template. Keys match the --attack argument and the data filenames.
PROMPTS = {
    "read_preposition": prompt_read_preposition,
    "read_tense": prompt_read_tense,
    "read_spelling": prompt_read_spelling,
    "read_reordering": prompt_read_reordering,
    "adeq_entity_1to1": prompt_adeq_entity_1to1,
    "adeq_polarity_swap": prompt_adeq_antonym_swap,
    "adeq_numerical": prompt_adeq_numerical,
    "adeq_coreference": prompt_adeq_coreference,
}

ATTACKS = list(PROMPTS)