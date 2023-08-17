import random

def generate_dataset(model, NUM_PROMPTS, TOTAL_TYPES):
    objects = [
    "perfume",
    "scissors",
    "drum",
    "trumpet",
    "phone",
    "football",
    "token",
    "bracelet",
    "badge",
    "novel",
    "pillow",
    "coffee",
    "skirt",
    "balloon",
    "photo",
    "plate",
    "headphones",
    "flask",
    "menu",
    "compass",
    "belt",
    "wallet",
    "pen",
    "mask",
    "ticket",
    "suitcase",
    "sunscreen",
    "letter",
    "torch",
    "cocktail",
    "spoon",
    "comb",
    "shirt",
    "coin",
    "cable",
    "button",
    "recorder",
    "frame",
    "key",
    "card",
    "canvas",
    "packet",
    "bowl",
    "receipt",
    "pan",
    "report",
    "book",
    "cap",
    "charger",
    "rake",
    "fork",
    "map",
    "soap",
    "cash",
    "whistle",
    "rope",
    "violin",
    "scale",
    "diary",
    "ruler",
    "mouse",
    "toy",
    "cd",
    "dress",
    "shampoo",
    "flashlight",
    "newspaper",
    "puzzle",
    "tripod",
    "brush",
    "cane",
    "whisk",
    "tablet",
    "purse",
    "paper",
    "vinyl",
    "camera",
    "guitar",
    "necklace",
    "mirror",
    "cup",
    "cloth",
    "flag",
    "socks",
    "shovel",
    "cooler",
    "hammer",
    "shoes",
    "chalk",
    "wrench",
    "towel",
    "glove",
    "speaker",
    "remote",
    "leash",
    "magazine",
    "notebook",
    "candle",
    "feather",
    "gloves",
    "mascara",
    "charcoal",
    "pills",
    "laptop",
    "pamphlet",
    "knife",
    "kettle",
    "scarf",
    "tie",
    "goggles",
    "fins",
    "lipstick",
    "shorts",
    "joystick",
    "bookmark",
    "microphone",
    "hat",
    "pants",
    "umbrella",
    "harness",
    "roller",
    "blanket",
    "folder",
    "bag",
    "crate",
    "pot",
    "watch",
    "mug",
    "sandwich",
    "yarn",
    "ring",
    "backpack",
    "glasses",
    "pencil",
    "broom",
    "baseball",
    "basket",
    "loaf",
    "coins",
    "bakery",
    "tape",
    "helmet",
    "bible",
    "jacket"
    ]

    names = [
    " Sebastian",
    " Jack",
    " Jeremiah",
    " Ellie",
    " Sean",
    " William",
    " Caroline",
    " Cooper",
    " Xavier",
    " Ian",
    " Mark",
    " Brian",
    " Carter",
    " Nicholas",
    " Peyton",
    " Luke",
    " Alexis",
    " Ted",
    " Jan",
    " Ty",
    " Jen",
    " Sophie",
    " Kelly",
    " Claire",
    " Leo",
    " Nolan",
    " Kyle",
    " Ashley",
    " Samantha",
    " Avery",
    " Jackson",
    " Hudson",
    " Rebecca",
    " Robert",
    " Joshua",
    " Olivia",
    " Reagan",
    " Lauren",
    " Chris",
    " Chelsea",
    " Deb",
    " Chloe",
    " Madison",
    " Kent",
    " Thomas",
    " Oliver",
    " Dylan",
    " Ann",
    " Audrey",
    " Greg",
    " Henry",
    " Emma",
    " Josh",
    " Mary",
    " Daniel",
    " Carl",
    " Scarlett",
    " Ethan",
    " Levi",
    " Eli",
    " James",
    " Patrick",
    " Isaac",
    " Brooke",
    " Alexa",
    " Eleanor",
    " Anthony",
    " Logan",
    " Damian",
    " Jordan",
    " Tyler",
    " Haley",
    " Isabel",
    " Alan",
    " Lucas",
    " Dave",
    " Susan",
    " Joseph",
    " Brad",
    " Joe",
    " Vincent",
    " Maya",
    " Will",
    " Jessica",
    " Sophia",
    " Angel",
    " Steve",
    " Benjamin",
    " Eric",
    " Cole",
    " Justin",
    " Amy",
    " Nora",
    " Seth",
    " Anna",
    " Stella",
    " Frank",
    " Larry",
    " Alexandra",
    " Ken",
    " Lucy",
    " Katherine",
    " Leah",
    " Adrian",
    " David",
    " Liam",
    " Christian",
    " John",
    " Nathaniel",
    " Andrea",
    " Laura",
    " Kim",
    " Kevin",
    " Colin",
    " Marcus",
    " Emily",
    " Sarah",
    " Steven",
    " Eva",
    " Richard",
    " Faith",
    " Amelia",
    " Harper",
    " Keith",
    " Ross",
    " Megan",
    " Brooklyn",
    " Tom",
    " Grant",
    " Savannah",
    " Riley",
    " Julia",
    " Piper",
    " Wyatt",
    " Jake",
    " Nathan",
    " Nick",
    " Blake",
    " Ryan",
    " Jason",
    " Chase",]

    places = [
    "swamp",
    "school",
    "volcano",
    "hotel",
    "subway",
    "arcade",
    "library",
    "island",
    "convent",
    "pool",
    "mall",
    "prison",
    "quarry",
    "temple",
    "ruins",
    "factory",
    "zoo",
    "mansion",
    "tavern",
    "planet",
    "forest",
    "airport",
    "pharmacy",
    "church",
    "park",
    "delta",
    "mosque",
    "valley",
    "casino",
    "pyramid",
    "aquarium",
    "castle",
    "ranch",
    "clinic",
    "theater",
    "gym",
    "studio",
    "station",
    "palace",
    "stadium",
    "museum",
    "plateau",
    "home",
    "resort",
    "garage",
    "reef",
    "lounge",
    "chapel",
    "canyon",
    "brewery",
    "market",
    "jungle",
    "office",
    "cottage",
    "street",
    "gallery",
    "landfill",
    "glacier",
    "barracks",
    "bakery",
    "synagogue",
    "jersey",
    "plaza",
    "garden",
    "cafe",
    "cinema",
    "beach",
    "harbor",
    "circus",
    "bridge",
    "monastery",
    "desert",
    "tunnel",
    "motel",
    "fortress"
    ]
    # %%
    
    one_token_no_space_names = [] # names for which the no-space, lower case version of it is also a single token
    for name in names:
        shortened_name = name.lower().replace(" ", "")
        if len(model.to_str_tokens(shortened_name, prepend_bos=False)) == 1:
            one_token_no_space_names.append(name)

    lower_case_still_same_name = []
    for name in names:
        lower_name = name.lower()
        if len(model.to_str_tokens(lower_name, prepend_bos=False)) == 1:
            lower_case_still_same_name.append(name)

    one_token_objects = []
    for obj in objects:
        longer_obj = " " + obj
        if len(model.to_str_tokens(longer_obj, prepend_bos=False)) == 1:
            one_token_objects.append(longer_obj)

    one_token_places = []
    for place in places:
        longer_place = " " + place
        if len(model.to_str_tokens(longer_place, prepend_bos=False)) == 1:
            one_token_places.append(longer_place)

    one_token_names = []

    for name in names:
        longer_name = name
        if len(model.to_str_tokens(longer_name, prepend_bos=False)) == 1:
            one_token_names.append(longer_name)

    # %%
    type_names = ["IOI", "Induction", "Handle Filling", "Mr/Mrs Implication"]
    IOI_template = "When{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"

    # test variants of induction
    Induction_templates = ["So,{name_A} ensured the{object} was brought out of the{place}. The",
                        "Thus,{name_A} allowed the{object} to enter into the grand{place}. The",
                        "Now,{name_A} guaranteed the{object} had fallen into the nearby{place}. The"]

    handle_templates = ["To learn more about the{object}, we spoke with{name_A}{name_B} (@{modified_name_A}",
                    "To build our own{object}, we got help from{name_A}{name_B} (@{modified_name_A}"]

    ignore_mr_templates = ["We talked to{name_A}{name_B}, who held the{object}. As Mr.",
                        "We spoke with{name_A}{name_B}, who held the{object}. As Mrs."]
    # %%
    ioi_string = IOI_template.format(
        name_A = one_token_names[0],
        name_B = one_token_names[1],
        name_C = one_token_names[0],
        place = one_token_places[0],
        object = one_token_objects[0]
    )
    induction_string_A = Induction_templates[0].format(
        name_A = one_token_names[1],
        object = one_token_objects[2],
        place = one_token_places[5]
    )
    induction_string_B = Induction_templates[1].format(
        name_A = one_token_names[4],
        object = one_token_objects[1],
        place = one_token_places[3]
    )
    induction_string_C = Induction_templates[2].format(
        name_A = one_token_names[2],
        object = one_token_objects[3],
        place = one_token_places[2]
    )

    handle_string_A = handle_templates[0].format(
        object = one_token_objects[0],
        name_A = one_token_no_space_names[4],
        name_B = one_token_no_space_names[1],
        # modified name is name_A but all lower case and now leading space
        modified_name_A = one_token_no_space_names[4].lower().replace(" ", "")
    )

    handle_string_B = handle_templates[1].format(
        object = one_token_objects[1],
        name_A = one_token_no_space_names[2],
        name_B = one_token_no_space_names[3],
        # modified name is name_A but all lower case and now leading space
        modified_name_A = one_token_no_space_names[2].lower().replace(" ", "")
    )

    ignore_mr_string = ignore_mr_templates[0].format(
        name_A = one_token_names[0],
        name_B = one_token_names[1],
        object = one_token_objects[0]
    )

    ignore_mrs_string = ignore_mr_templates[1].format(
        name_A = one_token_names[0],
        name_B = one_token_names[1],
        object = one_token_objects[0]
    )



    # %%
    print(ioi_string)
    print(induction_string_A)
    print(induction_string_B)
    print(induction_string_C)
    print(handle_string_A)
    print(handle_string_B)
    print(ignore_mr_string)
    print(ignore_mrs_string)


    print(model.to_tokens(ioi_string).shape)
    print(model.to_tokens(induction_string_A).shape)
    print(model.to_tokens(induction_string_B).shape)
    print(model.to_tokens(induction_string_C).shape)
    print(model.to_tokens(handle_string_A).shape)
    print(model.to_tokens(handle_string_B).shape)
    print(model.to_tokens(ignore_mr_string).shape)
    print(model.to_tokens(ignore_mrs_string).shape)

    print(model.to_str_tokens(ioi_string))
    print(model.to_str_tokens(induction_string_A))
    print(model.to_str_tokens(induction_string_B))
    print(model.to_str_tokens(induction_string_C))
    print(model.to_str_tokens(handle_string_A))
    print(model.to_str_tokens(handle_string_B))
    print(model.to_str_tokens(ignore_mr_string))
    print(model.to_str_tokens(ignore_mrs_string))
    # %%
    # TOTAL_TYPES = 4
    # NUM_PROMPTS = 20 * 6 * TOTAL_TYPES
    PROMPTS_PER_TYPE = int(NUM_PROMPTS / TOTAL_TYPES)


    PROMPTS = []
    CORRUPTED_PROMPTS = []
    ANSWERS = []
    INCORRECT_ANSWERS = []
    # generate IOI prompts
    for i in range(PROMPTS_PER_TYPE):
        name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]
        # generate name B that is different than A
        name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_B == name_A:
            name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]

        #generate name C diff from A and B
        name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_C == name_A or name_C == name_B:
            name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]

        # generate diff name D
        name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_D == name_A or name_D == name_B or name_D == name_C:
            name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]

        place_A = one_token_places[random.randint(0, len(one_token_places) - 1)]
        object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]

        PROMPTS.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        CORRUPTED_PROMPTS.append(IOI_template.format(
            name_A = name_C,
            name_B = name_D,
            name_C = name_C,
            place = place_A,
            object = object_A
        ))

        ANSWERS.append(name_B)
        INCORRECT_ANSWERS.append(name_A)


    # generate induction prompts
    INDUCTION_TYPES = 3
    for i in range(int(PROMPTS_PER_TYPE / INDUCTION_TYPES)):
        for prompt_type in range(INDUCTION_TYPES):
            name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]

            object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
            object_B = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
            while object_B == object_A:
                object_B = one_token_objects[random.randint(0, len(one_token_objects) - 1)]


            place_A = one_token_places[random.randint(0, len(one_token_places) - 1)]
            PROMPTS.append(Induction_templates[prompt_type].format(
                name_A = name_A,
                object = object_A,
                place = place_A
            ))

            CORRUPTED_PROMPTS.append(Induction_templates[prompt_type].format(
                name_A = name_A,
                object = object_B,
                place = place_A
            ))

            ANSWERS.append(object_A)
            #INCORRECT_ANSWERS.append(object_B) # if we want incorrect to be some other object
            INCORRECT_ANSWERS.append(name_A) # if we want incorrect to be name in prompt

    # generate handle prompts
    HANDLE_TYPES = 2
    for i in range(int(PROMPTS_PER_TYPE / HANDLE_TYPES)):
        for prompt_type in range(HANDLE_TYPES):
            name_A = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            name_B = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            while name_B == name_A:
                name_B = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]

            name_C = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            while name_C == name_A or name_C == name_B:
                name_C = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            name_D = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            while name_D == name_A or name_D == name_B or name_D == name_C:
                name_D = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]

        

            object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
            PROMPTS.append(handle_templates[prompt_type].format(
                object = object_A,
                name_A = name_A,
                name_B = name_B,
                # modified name is name_A but all lower case and now leading space
                modified_name_A = name_A.lower().replace(" ", "")
            ))
        
        
            CORRUPTED_PROMPTS.append(handle_templates[prompt_type].format(
                object = object_A,
                name_A = name_C,
                name_B = name_D,
                # modified name is name_A but all lower case and now leading space
                modified_name_A = name_C.lower().replace(" ", "")
            ))


            ANSWERS.append(name_B.lower().replace(" ", ""))
            INCORRECT_ANSWERS.append(name_A.lower().replace(" ", ""))


    # generate ignore mr prompts
    IGNORE_MR_TYPES = 2
    for i in range(int(PROMPTS_PER_TYPE / IGNORE_MR_TYPES)):
        for prompt_type in range(IGNORE_MR_TYPES):
            name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]
            name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
            while name_B == name_A:
                name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
            name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]
            while name_C == name_A or name_C == name_B:
                name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]
            name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]
            while name_D == name_A or name_D == name_B or name_D == name_C:
                name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]

            object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]

            PROMPTS.append(ignore_mr_templates[prompt_type].format(
                name_A = name_A,
                name_B = name_B,
                object = object_A
            ))
            CORRUPTED_PROMPTS.append(ignore_mr_templates[prompt_type].format(
                name_A = name_C,
                name_B = name_D,
                object = object_A
            ))

            ANSWERS.append(name_B)
            INCORRECT_ANSWERS.append(name_A)
    return PROMPTS, CORRUPTED_PROMPTS, ANSWERS, INCORRECT_ANSWERS, type_names



def generate_four_IOI_types(model, NUM_PROMPTS_PER_TYPE):

    objects = [
    "perfume",
    "scissors",
    "drum",
    "trumpet",
    "phone",
    "football",
    "token",
    "bracelet",
    "badge",
    "novel",
    "pillow",
    "coffee",
    "skirt",
    "balloon",
    "photo",
    "plate",
    "headphones",
    "flask",
    "menu",
    "compass",
    "belt",
    "wallet",
    "pen",
    "mask",
    "ticket",
    "suitcase",
    "sunscreen",
    "letter",
    "torch",
    "cocktail",
    "spoon",
    "comb",
    "shirt",
    "coin",
    "cable",
    "button",
    "recorder",
    "frame",
    "key",
    "card",
    "canvas",
    "packet",
    "bowl",
    "receipt",
    "pan",
    "report",
    "book",
    "cap",
    "charger",
    "rake",
    "fork",
    "map",
    "soap",
    "cash",
    "whistle",
    "rope",
    "violin",
    "scale",
    "diary",
    "ruler",
    "mouse",
    "toy",
    "cd",
    "dress",
    "shampoo",
    "flashlight",
    "newspaper",
    "puzzle",
    "tripod",
    "brush",
    "cane",
    "whisk",
    "tablet",
    "purse",
    "paper",
    "vinyl",
    "camera",
    "guitar",
    "necklace",
    "mirror",
    "cup",
    "cloth",
    "flag",
    "socks",
    "shovel",
    "cooler",
    "hammer",
    "shoes",
    "chalk",
    "wrench",
    "towel",
    "glove",
    "speaker",
    "remote",
    "leash",
    "magazine",
    "notebook",
    "candle",
    "feather",
    "gloves",
    "mascara",
    "charcoal",
    "pills",
    "laptop",
    "pamphlet",
    "knife",
    "kettle",
    "scarf",
    "tie",
    "goggles",
    "fins",
    "lipstick",
    "shorts",
    "joystick",
    "bookmark",
    "microphone",
    "hat",
    "pants",
    "umbrella",
    "harness",
    "roller",
    "blanket",
    "folder",
    "bag",
    "crate",
    "pot",
    "watch",
    "mug",
    "sandwich",
    "yarn",
    "ring",
    "backpack",
    "glasses",
    "pencil",
    "broom",
    "baseball",
    "basket",
    "loaf",
    "coins",
    "bakery",
    "tape",
    "helmet",
    "bible",
    "jacket"
    ]

    names = [
    " Sebastian",
    " Jack",
    " Jeremiah",
    " Ellie",
    " Sean",
    " William",
    " Caroline",
    " Cooper",
    " Xavier",
    " Ian",
    " Mark",
    " Brian",
    " Carter",
    " Nicholas",
    " Peyton",
    " Luke",
    " Alexis",
    " Ted",
    " Jan",
    " Ty",
    " Jen",
    " Sophie",
    " Kelly",
    " Claire",
    " Leo",
    " Nolan",
    " Kyle",
    " Ashley",
    " Samantha",
    " Avery",
    " Jackson",
    " Hudson",
    " Rebecca",
    " Robert",
    " Joshua",
    " Olivia",
    " Reagan",
    " Lauren",
    " Chris",
    " Chelsea",
    " Deb",
    " Chloe",
    " Madison",
    " Kent",
    " Thomas",
    " Oliver",
    " Dylan",
    " Ann",
    " Audrey",
    " Greg",
    " Henry",
    " Emma",
    " Josh",
    " Mary",
    " Daniel",
    " Carl",
    " Scarlett",
    " Ethan",
    " Levi",
    " Eli",
    " James",
    " Patrick",
    " Isaac",
    " Brooke",
    " Alexa",
    " Eleanor",
    " Anthony",
    " Logan",
    " Damian",
    " Jordan",
    " Tyler",
    " Haley",
    " Isabel",
    " Alan",
    " Lucas",
    " Dave",
    " Susan",
    " Joseph",
    " Brad",
    " Joe",
    " Vincent",
    " Maya",
    " Will",
    " Jessica",
    " Sophia",
    " Angel",
    " Steve",
    " Benjamin",
    " Eric",
    " Cole",
    " Justin",
    " Amy",
    " Nora",
    " Seth",
    " Anna",
    " Stella",
    " Frank",
    " Larry",
    " Alexandra",
    " Ken",
    " Lucy",
    " Katherine",
    " Leah",
    " Adrian",
    " David",
    " Liam",
    " Christian",
    " John",
    " Nathaniel",
    " Andrea",
    " Laura",
    " Kim",
    " Kevin",
    " Colin",
    " Marcus",
    " Emily",
    " Sarah",
    " Steven",
    " Eva",
    " Richard",
    " Faith",
    " Amelia",
    " Harper",
    " Keith",
    " Ross",
    " Megan",
    " Brooklyn",
    " Tom",
    " Grant",
    " Savannah",
    " Riley",
    " Julia",
    " Piper",
    " Wyatt",
    " Jake",
    " Nathan",
    " Nick",
    " Blake",
    " Ryan",
    " Jason",
    " Chase",]

    places = [
    "swamp",
    "school",
    "volcano",
    "hotel",
    "subway",
    "arcade",
    "library",
    "island",
    "convent",
    "pool",
    "mall",
    "prison",
    "quarry",
    "temple",
    "ruins",
    "factory",
    "zoo",
    "mansion",
    "tavern",
    "planet",
    "forest",
    "airport",
    "pharmacy",
    "church",
    "park",
    "delta",
    "mosque",
    "valley",
    "casino",
    "pyramid",
    "aquarium",
    "castle",
    "ranch",
    "clinic",
    "theater",
    "gym",
    "studio",
    "station",
    "palace",
    "stadium",
    "museum",
    "plateau",
    "home",
    "resort",
    "garage",
    "reef",
    "lounge",
    "chapel",
    "canyon",
    "brewery",
    "market",
    "jungle",
    "office",
    "cottage",
    "street",
    "gallery",
    "landfill",
    "glacier",
    "barracks",
    "bakery",
    "synagogue",
    "jersey",
    "plaza",
    "garden",
    "cafe",
    "cinema",
    "beach",
    "harbor",
    "circus",
    "bridge",
    "monastery",
    "desert",
    "tunnel",
    "motel",
    "fortress"
    ]


    one_token_no_space_names = [] # names for which the no-space, lower case version of it is also a single token
    for name in names:
        shortened_name = name.lower().replace(" ", "")
        if len(model.to_str_tokens(shortened_name, prepend_bos=False)) == 1:
            one_token_no_space_names.append(name)

    lower_case_still_same_name = []
    for name in names:
        lower_name = name.lower()
        if len(model.to_str_tokens(lower_name, prepend_bos=False)) == 1:
            lower_case_still_same_name.append(name)

    one_token_objects = []
    for obj in objects:
        longer_obj = " " + obj
        if len(model.to_str_tokens(longer_obj, prepend_bos=False)) == 1:
            one_token_objects.append(longer_obj)

    one_token_places = []
    for place in places:
        longer_place = " " + place
        if len(model.to_str_tokens(longer_place, prepend_bos=False)) == 1:
            one_token_places.append(longer_place)

    one_token_names = []

    for name in names:
        longer_name = name
        if len(model.to_str_tokens(longer_name, prepend_bos=False)) == 1:
            one_token_names.append(longer_name)


    IOI_template = "When{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"

    names = []
    ABB = []
    ABA = []
    BAA = []
    BAB = []

    for i in range(NUM_PROMPTS_PER_TYPE):
        name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]
        # generate name B that is different than A
        name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_B == name_A:
            name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]


        names.append((name_A, name_B))
        place_A = one_token_places[random.randint(0, len(one_token_places) - 1)]
        object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]

        ABB.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))

        ABA.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        BAA.append(IOI_template.format(
            name_A = name_B,
            name_B = name_A,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        BAB.append(IOI_template.format(
            name_A = name_B,
            name_B = name_A,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))
    
    return ABB, ABA, BAA, BAB, names






def generate_four_IOI_types_plus_offset_intro(model, NUM_PROMPTS_PER_TYPE):


    
    objects = [
    "perfume",
    "scissors",
    "drum",
    "trumpet",
    "phone",
    "football",
    "token",
    "bracelet",
    "badge",
    "novel",
    "pillow",
    "coffee",
    "skirt",
    "balloon",
    "photo",
    "plate",
    "headphones",
    "flask",
    "menu",
    "compass",
    "belt",
    "wallet",
    "pen",
    "mask",
    "ticket",
    "suitcase",
    "sunscreen",
    "letter",
    "torch",
    "cocktail",
    "spoon",
    "comb",
    "shirt",
    "coin",
    "cable",
    "button",
    "recorder",
    "frame",
    "key",
    "card",
    "canvas",
    "packet",
    "bowl",
    "receipt",
    "pan",
    "report",
    "book",
    "cap",
    "charger",
    "rake",
    "fork",
    "map",
    "soap",
    "cash",
    "whistle",
    "rope",
    "violin",
    "scale",
    "diary",
    "ruler",
    "mouse",
    "toy",
    "cd",
    "dress",
    "shampoo",
    "flashlight",
    "newspaper",
    "puzzle",
    "tripod",
    "brush",
    "cane",
    "whisk",
    "tablet",
    "purse",
    "paper",
    "vinyl",
    "camera",
    "guitar",
    "necklace",
    "mirror",
    "cup",
    "cloth",
    "flag",
    "socks",
    "shovel",
    "cooler",
    "hammer",
    "shoes",
    "chalk",
    "wrench",
    "towel",
    "glove",
    "speaker",
    "remote",
    "leash",
    "magazine",
    "notebook",
    "candle",
    "feather",
    "gloves",
    "mascara",
    "charcoal",
    "pills",
    "laptop",
    "pamphlet",
    "knife",
    "kettle",
    "scarf",
    "tie",
    "goggles",
    "fins",
    "lipstick",
    "shorts",
    "joystick",
    "bookmark",
    "microphone",
    "hat",
    "pants",
    "umbrella",
    "harness",
    "roller",
    "blanket",
    "folder",
    "bag",
    "crate",
    "pot",
    "watch",
    "mug",
    "sandwich",
    "yarn",
    "ring",
    "backpack",
    "glasses",
    "pencil",
    "broom",
    "baseball",
    "basket",
    "loaf",
    "coins",
    "bakery",
    "tape",
    "helmet",
    "bible",
    "jacket"
    ]

    names = [
    " Sebastian",
    " Jack",
    " Jeremiah",
    " Ellie",
    " Sean",
    " William",
    " Caroline",
    " Cooper",
    " Xavier",
    " Ian",
    " Mark",
    " Brian",
    " Carter",
    " Nicholas",
    " Peyton",
    " Luke",
    " Alexis",
    " Ted",
    " Jan",
    " Ty",
    " Jen",
    " Sophie",
    " Kelly",
    " Claire",
    " Leo",
    " Nolan",
    " Kyle",
    " Ashley",
    " Samantha",
    " Avery",
    " Jackson",
    " Hudson",
    " Rebecca",
    " Robert",
    " Joshua",
    " Olivia",
    " Reagan",
    " Lauren",
    " Chris",
    " Chelsea",
    " Deb",
    " Chloe",
    " Madison",
    " Kent",
    " Thomas",
    " Oliver",
    " Dylan",
    " Ann",
    " Audrey",
    " Greg",
    " Henry",
    " Emma",
    " Josh",
    " Mary",
    " Daniel",
    " Carl",
    " Scarlett",
    " Ethan",
    " Levi",
    " Eli",
    " James",
    " Patrick",
    " Isaac",
    " Brooke",
    " Alexa",
    " Eleanor",
    " Anthony",
    " Logan",
    " Damian",
    " Jordan",
    " Tyler",
    " Haley",
    " Isabel",
    " Alan",
    " Lucas",
    " Dave",
    " Susan",
    " Joseph",
    " Brad",
    " Joe",
    " Vincent",
    " Maya",
    " Will",
    " Jessica",
    " Sophia",
    " Angel",
    " Steve",
    " Benjamin",
    " Eric",
    " Cole",
    " Justin",
    " Amy",
    " Nora",
    " Seth",
    " Anna",
    " Stella",
    " Frank",
    " Larry",
    " Alexandra",
    " Ken",
    " Lucy",
    " Katherine",
    " Leah",
    " Adrian",
    " David",
    " Liam",
    " Christian",
    " John",
    " Nathaniel",
    " Andrea",
    " Laura",
    " Kim",
    " Kevin",
    " Colin",
    " Marcus",
    " Emily",
    " Sarah",
    " Steven",
    " Eva",
    " Richard",
    " Faith",
    " Amelia",
    " Harper",
    " Keith",
    " Ross",
    " Megan",
    " Brooklyn",
    " Tom",
    " Grant",
    " Savannah",
    " Riley",
    " Julia",
    " Piper",
    " Wyatt",
    " Jake",
    " Nathan",
    " Nick",
    " Blake",
    " Ryan",
    " Jason",
    " Chase",]

    places = [
    "swamp",
    "school",
    "volcano",
    "hotel",
    "subway",
    "arcade",
    "library",
    "island",
    "convent",
    "pool",
    "mall",
    "prison",
    "quarry",
    "temple",
    "ruins",
    "factory",
    "zoo",
    "mansion",
    "tavern",
    "planet",
    "forest",
    "airport",
    "pharmacy",
    "church",
    "park",
    "delta",
    "mosque",
    "valley",
    "casino",
    "pyramid",
    "aquarium",
    "castle",
    "ranch",
    "clinic",
    "theater",
    "gym",
    "studio",
    "station",
    "palace",
    "stadium",
    "museum",
    "plateau",
    "home",
    "resort",
    "garage",
    "reef",
    "lounge",
    "chapel",
    "canyon",
    "brewery",
    "market",
    "jungle",
    "office",
    "cottage",
    "street",
    "gallery",
    "landfill",
    "glacier",
    "barracks",
    "bakery",
    "synagogue",
    "jersey",
    "plaza",
    "garden",
    "cafe",
    "cinema",
    "beach",
    "harbor",
    "circus",
    "bridge",
    "monastery",
    "desert",
    "tunnel",
    "motel",
    "fortress"
    ]

    prefixs = [
        "Yesterday",
        "Today",
        "Clearly",
        "Sadly",
        "Basically",
        "Hopefully",
        "Then",  
        "For" ]
    


    one_token_no_space_names = [] # names for which the no-space, lower case version of it is also a single token
    
    for name in names:
        shortened_name = name.lower().replace(" ", "")
        if len(model.to_str_tokens(shortened_name, prepend_bos=False)) == 1:
            one_token_no_space_names.append(name)

    lower_case_still_same_name = []
    for name in names:
        lower_name = name.lower()
        if len(model.to_str_tokens(lower_name, prepend_bos=False)) == 1:
            lower_case_still_same_name.append(name)

    one_token_objects = []
    for obj in objects:
        longer_obj = " " + obj
        if len(model.to_str_tokens(longer_obj, prepend_bos=False)) == 1:
            one_token_objects.append(longer_obj)

    one_token_places = []
    for place in places:
        longer_place = " " + place
        if len(model.to_str_tokens(longer_place, prepend_bos=False)) == 1:
            one_token_places.append(longer_place)

    one_token_names = []

    for name in names:
        longer_name = name
        if len(model.to_str_tokens(longer_name, prepend_bos=False)) == 1:
            one_token_names.append(longer_name)


    IOI_template = "When{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"
    ONE_WORD_template = "{prefix} when{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"

    names = []
    ABB = []
    ABA = []
    BAA = []
    BAB = []
    
    # intro with one word
    ONE_WORD_ABB = []
    ONE_WORD_ABA = []
    ONE_WORD_BAA = []
    ONE_WORD_BAB = []


    for i in range(NUM_PROMPTS_PER_TYPE):
        name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]
        # generate name B that is different than A
        name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_B == name_A:
            name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]


        names.append((name_A, name_B))
        place_A = one_token_places[random.randint(0, len(one_token_places) - 1)]
        object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
        prefix_A = prefixs[random.randint(0, len(prefixs) - 1)]

        ABB.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_ABB.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_A,
            name_B = name_B,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))
        

        ABA.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_ABA.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))


        BAA.append(IOI_template.format(
            name_A = name_B,
            name_B = name_A,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_BAA.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_B,
            name_B = name_A,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))


        BAB.append(IOI_template.format(
            name_A = name_B,
            name_B = name_A,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_BAB.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_B,
            name_B = name_A,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))
    
    
    return ABB, ABA, BAA, BAB, names, ONE_WORD_ABB, ONE_WORD_ABA, ONE_WORD_BAA, ONE_WORD_BAB






def generate_four_IOI_types_plus_offset_intro_AND_intro_name(model, NUM_PROMPTS_PER_TYPE):

    
    objects = [
    "perfume",
    "scissors",
    "drum",
    "trumpet",
    "phone",
    "football",
    "token",
    "bracelet",
    "badge",
    "novel",
    "pillow",
    "coffee",
    "skirt",
    "balloon",
    "photo",
    "plate",
    "headphones",
    "flask",
    "menu",
    "compass",
    "belt",
    "wallet",
    "pen",
    "mask",
    "ticket",
    "suitcase",
    "sunscreen",
    "letter",
    "torch",
    "cocktail",
    "spoon",
    "comb",
    "shirt",
    "coin",
    "cable",
    "button",
    "recorder",
    "frame",
    "key",
    "card",
    "canvas",
    "packet",
    "bowl",
    "receipt",
    "pan",
    "report",
    "book",
    "cap",
    "charger",
    "rake",
    "fork",
    "map",
    "soap",
    "cash",
    "whistle",
    "rope",
    "violin",
    "scale",
    "diary",
    "ruler",
    "mouse",
    "toy",
    "cd",
    "dress",
    "shampoo",
    "flashlight",
    "newspaper",
    "puzzle",
    "tripod",
    "brush",
    "cane",
    "whisk",
    "tablet",
    "purse",
    "paper",
    "vinyl",
    "camera",
    "guitar",
    "necklace",
    "mirror",
    "cup",
    "cloth",
    "flag",
    "socks",
    "shovel",
    "cooler",
    "hammer",
    "shoes",
    "chalk",
    "wrench",
    "towel",
    "glove",
    "speaker",
    "remote",
    "leash",
    "magazine",
    "notebook",
    "candle",
    "feather",
    "gloves",
    "mascara",
    "charcoal",
    "pills",
    "laptop",
    "pamphlet",
    "knife",
    "kettle",
    "scarf",
    "tie",
    "goggles",
    "fins",
    "lipstick",
    "shorts",
    "joystick",
    "bookmark",
    "microphone",
    "hat",
    "pants",
    "umbrella",
    "harness",
    "roller",
    "blanket",
    "folder",
    "bag",
    "crate",
    "pot",
    "watch",
    "mug",
    "sandwich",
    "yarn",
    "ring",
    "backpack",
    "glasses",
    "pencil",
    "broom",
    "baseball",
    "basket",
    "loaf",
    "coins",
    "bakery",
    "tape",
    "helmet",
    "bible",
    "jacket"
    ]

    names = [
    " Sebastian",
    " Jack",
    " Jeremiah",
    " Ellie",
    " Sean",
    " William",
    " Caroline",
    " Cooper",
    " Xavier",
    " Ian",
    " Mark",
    " Brian",
    " Carter",
    " Nicholas",
    " Peyton",
    " Luke",
    " Alexis",
    " Ted",
    " Jan",
    " Ty",
    " Jen",
    " Sophie",
    " Kelly",
    " Claire",
    " Leo",
    " Nolan",
    " Kyle",
    " Ashley",
    " Samantha",
    " Avery",
    " Jackson",
    " Hudson",
    " Rebecca",
    " Robert",
    " Joshua",
    " Olivia",
    " Reagan",
    " Lauren",
    " Chris",
    " Chelsea",
    " Deb",
    " Chloe",
    " Madison",
    " Kent",
    " Thomas",
    " Oliver",
    " Dylan",
    " Ann",
    " Audrey",
    " Greg",
    " Henry",
    " Emma",
    " Josh",
    " Mary",
    " Daniel",
    " Carl",
    " Scarlett",
    " Ethan",
    " Levi",
    " Eli",
    " James",
    " Patrick",
    " Isaac",
    " Brooke",
    " Alexa",
    " Eleanor",
    " Anthony",
    " Logan",
    " Damian",
    " Jordan",
    " Tyler",
    " Haley",
    " Isabel",
    " Alan",
    " Lucas",
    " Dave",
    " Susan",
    " Joseph",
    " Brad",
    " Joe",
    " Vincent",
    " Maya",
    " Will",
    " Jessica",
    " Sophia",
    " Angel",
    " Steve",
    " Benjamin",
    " Eric",
    " Cole",
    " Justin",
    " Amy",
    " Nora",
    " Seth",
    " Anna",
    " Stella",
    " Frank",
    " Larry",
    " Alexandra",
    " Ken",
    " Lucy",
    " Katherine",
    " Leah",
    " Adrian",
    " David",
    " Liam",
    " Christian",
    " John",
    " Nathaniel",
    " Andrea",
    " Laura",
    " Kim",
    " Kevin",
    " Colin",
    " Marcus",
    " Emily",
    " Sarah",
    " Steven",
    " Eva",
    " Richard",
    " Faith",
    " Amelia",
    " Harper",
    " Keith",
    " Ross",
    " Megan",
    " Brooklyn",
    " Tom",
    " Grant",
    " Savannah",
    " Riley",
    " Julia",
    " Piper",
    " Wyatt",
    " Jake",
    " Nathan",
    " Nick",
    " Blake",
    " Ryan",
    " Jason",
    " Chase",]

    places = [
    "swamp",
    "school",
    "volcano",
    "hotel",
    "subway",
    "arcade",
    "library",
    "island",
    "convent",
    "pool",
    "mall",
    "prison",
    "quarry",
    "temple",
    "ruins",
    "factory",
    "zoo",
    "mansion",
    "tavern",
    "planet",
    "forest",
    "airport",
    "pharmacy",
    "church",
    "park",
    "delta",
    "mosque",
    "valley",
    "casino",
    "pyramid",
    "aquarium",
    "castle",
    "ranch",
    "clinic",
    "theater",
    "gym",
    "studio",
    "station",
    "palace",
    "stadium",
    "museum",
    "plateau",
    "home",
    "resort",
    "garage",
    "reef",
    "lounge",
    "chapel",
    "canyon",
    "brewery",
    "market",
    "jungle",
    "office",
    "cottage",
    "street",
    "gallery",
    "landfill",
    "glacier",
    "barracks",
    "bakery",
    "synagogue",
    "jersey",
    "plaza",
    "garden",
    "cafe",
    "cinema",
    "beach",
    "harbor",
    "circus",
    "bridge",
    "monastery",
    "desert",
    "tunnel",
    "motel",
    "fortress"
    ]

    prefixs = [
        "Yesterday",
        "Today",
        "Clearly",
        "Sadly",
        "Basically",
        "Hopefully",
        "Then",  
        "For" ]
    


    one_token_no_space_names = [] # names for which the no-space, lower case version of it is also a single token
    
    for name in names:
        shortened_name = name.lower().replace(" ", "")
        if len(model.to_str_tokens(shortened_name, prepend_bos=False)) == 1:
            one_token_no_space_names.append(name)

    one_token_actual_no_space_names = []
    for name in names:
            shortened_name = name.replace(" ", "")
            if len(model.to_str_tokens(shortened_name, prepend_bos=False)) == 1:
                one_token_actual_no_space_names.append(name)


    lower_case_still_same_name = []
    for name in names:
        lower_name = name.lower()
        if len(model.to_str_tokens(lower_name, prepend_bos=False)) == 1:
            lower_case_still_same_name.append(name)

    one_token_objects = []
    for obj in objects:
        longer_obj = " " + obj
        if len(model.to_str_tokens(longer_obj, prepend_bos=False)) == 1:
            one_token_objects.append(longer_obj)

    one_token_places = []
    for place in places:
        longer_place = " " + place
        if len(model.to_str_tokens(longer_place, prepend_bos=False)) == 1:
            one_token_places.append(longer_place)

    one_token_names = []

    for name in names:
        longer_name = name
        if len(model.to_str_tokens(longer_name, prepend_bos=False)) == 1:
            one_token_names.append(longer_name)




    IOI_template = "When{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"
    ONE_WORD_template = "{prefix} when{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"
    PREFIX_NAME_template = "{name_P} when{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"

    names = []
    ABB = []
    ABA = []
    BAA = []
    BAB = []
    
    # intro with one word
    ONE_WORD_ABB = []
    ONE_WORD_ABA = []
    ONE_WORD_BAA = []
    ONE_WORD_BAB = []

    # intro with name
    PREFIX_NAME_ABB = []
    PREFIX_NAME_ABA = []
    PREFIX_NAME_BAA = []
    PREFIX_NAME_BAB = []




    for i in range(NUM_PROMPTS_PER_TYPE):
        name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]
        # generate name B that is different than A
        name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_B == name_A:
            name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]


        name_C = None # for the intro of PREFIX_NAME data
        while name_C == None:
            shortened_name_C = one_token_actual_no_space_names[random.randint(0, len(one_token_actual_no_space_names) - 1)]
            longer_name_C = " " + shortened_name_C
            if longer_name_C != name_A and longer_name_C != name_B:
                name_C = shortened_name_C
                    
        assert name_C != None


        names.append((name_A, name_B))
        place_A = one_token_places[random.randint(0, len(one_token_places) - 1)]
        object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
        prefix_A = prefixs[random.randint(0, len(prefixs) - 1)]

        ABB.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_ABB.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_A,
            name_B = name_B,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))


        PREFIX_NAME_ABB.append(PREFIX_NAME_template.format(
            name_P = name_C,
            name_A = name_A,
            name_B = name_B,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))
        

        ABA.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_ABA.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        PREFIX_NAME_ABA.append(PREFIX_NAME_template.format(
            name_P = name_C,
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))


        BAA.append(IOI_template.format(
            name_A = name_B,
            name_B = name_A,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_BAA.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_B,
            name_B = name_A,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        PREFIX_NAME_BAA.append(PREFIX_NAME_template.format(
            name_P = name_C,
            name_A = name_B,
            name_B = name_A,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))


        BAB.append(IOI_template.format(
            name_A = name_B,
            name_B = name_A,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))

        ONE_WORD_BAB.append(ONE_WORD_template.format(
            prefix = prefix_A,
            name_A = name_B,
            name_B = name_A,
            name_C = name_B,
            place = place_A,
            object = object_A
        ))
    
        PREFIX_NAME_BAB.append(PREFIX_NAME_template.format(
            name_P = name_C,
            name_A = name_B,
            name_B = name_A,
            name_C = name_B,
            place = place_A,
            object = object_A
        )) 
    return ABB, ABA, BAA, BAB, names, ONE_WORD_ABB, ONE_WORD_ABA, ONE_WORD_BAA, ONE_WORD_BAB, PREFIX_NAME_ABB, PREFIX_NAME_ABA, PREFIX_NAME_BAA, PREFIX_NAME_BAB