from transformers.models.auto.tokenization_auto import AutoTokenizer
from models.baseline_model.predict import collate

model_name = "jannahalka/nlp-project-baseline"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

batch_sentences = [
    "But if going up against determined and dug-in Separatist forces wasn't enough, the Jedi must also contend with a glory-seeking young Republic officer in their midst, Captain Kendal Ozzel.",
    "Windu and the three other Jedi Masters arrive at Chancellor Palpatine's office.",
    "With the shields down, Blue Squadron and Red Squadron led by Black Leader Poe Dameron commence their assault.",
    "Ren becomes so impressed with Rey that he tries to tempt her, complimenting her strength with the Force and offering to complete her training if she joins him.",
    "They steal First Order officer uniforms from an automated laundry room and make their way to a hyperspace tracker.",
    "The conversation reveals that Jax bribed Palpatine's physician to sabotage his remaining clones after the destruction of Byss.",
    "On the Death Star, in the middle of the evacuation, Luke carries his father's ravaged body to the foot of an Imperial shuttle's ramp.",
    "Using Grievous's electrostaff, he manages to destabilize the speeder, and it goes into a spin.",
    "Each girl is selected for her particular talents, but it will be up to Padmé to unite them as a group.",
    "Captain Roos Tarpals orders the Gungan Grand Army to activate their shield, which protects them from ranged attack.",
    "Obi-Wan Kenobi and Anakin Skywalker must stem the tide of the raging Clone Wars and forge a new bond as Jedi Knights.",
    "While the Congress of the Republic endlessly debates this alarming chain of events, the Supreme Chancellor has secretly dispatched two Jedi Knights, the guardians of peace and justice in the galaxy, to settle the conflict…",
    "As the battle rages on above the castle, the remaining First Order troops board their ships and retreat, and Han witnesses his son carrying an unconscious Rey away.",
]

def test_collate():
    collated_batch = collate(batch_sentences, tokenizer)
    input_ids = collated_batch["input_ids"]

    assert len(input_ids) == len(batch_sentences)
