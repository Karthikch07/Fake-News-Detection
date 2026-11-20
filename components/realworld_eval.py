"""Hand-written real-world evaluation set (never used for training).

Realistic headlines/snippets in the style people actually paste into the app,
covering several regions, topics and hoax styles. The model's accuracy on this
set is reported in the UI as "real-world accuracy", next to the (much more
optimistic) accuracy on the training dataset's own held-out test split.
"""

REAL = [
    "The Federal Reserve held interest rates steady on Wednesday, citing continued progress on inflation, according to a statement released after the two-day meeting.",
    "NASA's Artemis program delayed its next crewed lunar mission after engineers identified issues with the Orion capsule's heat shield, the agency said.",
    "The World Health Organization reported a rise in measles cases across Europe last year, urging countries to boost vaccination coverage.",
    "Apple reported quarterly revenue of $94.9 billion, up 6 percent from a year earlier, driven by strong iPhone sales in emerging markets.",
    "India's monsoon rainfall was 8% above average this season, the India Meteorological Department said on Monday.",
    "The European Parliament approved new rules on artificial intelligence, setting obligations for high-risk systems, after lawmakers voted 523 to 46.",
    "Officials in Japan issued a tsunami advisory after a magnitude 6.8 earthquake struck off the coast of Hokkaido, the Japan Meteorological Agency said.",
    "A study published in The Lancet found that regular physical activity was associated with a lower risk of cardiovascular disease among adults over 60.",
    "Shares in Tesla fell 4% in early trading after the company reported fewer vehicle deliveries than analysts had expected.",
    "The UK's Office for National Statistics said unemployment rose slightly to 4.3% in the three months to September.",
    "Police said two people were injured in a collision on the M25 on Tuesday morning and the motorway was partially closed for several hours.",
    "The Supreme Court agreed to hear a case on the limits of federal agencies' regulatory authority, with arguments expected early next year.",
    "Researchers at MIT developed a new battery chemistry that could reduce costs for grid storage, according to a paper in Nature Energy.",
    "Germany's parliament passed the budget for next year after weeks of negotiations among the coalition partners.",
    "The city council voted to expand the bus network, adding three new routes that will begin service in March, a spokesperson confirmed.",
    "The Reserve Bank of India kept the repo rate unchanged at 6.5% for the eighth consecutive meeting, Governor Shaktikanta Das said.",
    "ISRO successfully placed a weather satellite into orbit on Saturday, the space agency said in a statement.",
    "Kenya's health ministry said a cholera outbreak in two coastal counties had been contained after a vaccination drive reached 200,000 people.",
    "Brazil's statistics agency said the economy grew 0.9% in the second quarter, beating economists' forecasts.",
    "Australian authorities ordered evacuations in parts of New South Wales as floodwaters rose after days of heavy rain.",
    "The Indian cricket team won the third Test by 64 runs to take an unassailable 3-1 lead in the five-match series.",
    "Microsoft said it would invest $3 billion in cloud and AI infrastructure in India over the next two years.",
    "A new study of 12,000 children found that screen time above four hours a day was linked to poorer sleep, researchers reported in JAMA Pediatrics.",
    "Canada's wildfire agency said more than 400 fires were burning across the country, about half of them out of control.",
    "The European Central Bank cut its deposit rate by a quarter point, saying inflation was moving toward its 2% target.",
    "Volkswagen said it would recall about 30,000 vehicles in the United States over a faulty seat-belt sensor.",
    "The municipal corporation said water supply in several wards would be suspended for 12 hours on Thursday for pipeline repairs.",
    "Scientists sequenced the genome of an ancient wheat variety, which may help breeders develop drought-tolerant crops, the journal Nature reported.",
    "South Korea's exports rose for a fifth straight month in October, led by semiconductor shipments, trade ministry data showed.",
    "The airline cancelled dozens of flights on Monday after a technical outage affected its check-in systems, a company spokesperson said.",
]

# Style-swapped hard cases: true facts written casually, and false claims
# written in calm wire-service style. A model that judges writing style
# rather than facts gets these wrong, so they are reported separately to
# keep the headline accuracy honest.
HARD_REAL = [
    "omg the Eiffel Tower actually grows about 15 cm taller in summer because the metal expands in the heat!!",
    "Can't believe it but octopuses really do have three hearts and blue blood. nature is wild",
    "Honey found in ancient Egyptian tombs was still edible after 3,000 years. Archaeologists weren't kidding!",
    "Wow. Obama signed the Affordable Care Act into law back in 2010 and it is still the law today",
    "Crazy fact: Bananas are slightly radioactive because they contain potassium-40. Totally harmless though!",
    "So it turns out a day on Venus is longer than its whole year. Space is weird lol",
    "Breaking my brain: there are more possible chess games than atoms in the observable universe",
]

HARD_FAKE = [
    "The health ministry said on Thursday that the measles vaccine causes autism, according to a statement released by officials.",
    "NASA confirmed in a statement on Tuesday that the Great Wall of China is clearly visible to the naked eye from the Moon.",
    "The World Health Organization said on Monday that drinking two litres of hot water a day prevents all viral infections, officials told reporters.",
    "Scientists at the national laboratory said on Wednesday that humans use only 10 percent of their brains, according to a peer-reviewed study.",
    "The central bank announced on Friday that all currency notes will be withdrawn and replaced by gold coins next month, a spokesperson said.",
    "The European Space Agency said in a statement that the Sun will go dark for three days in December due to a planetary alignment.",
    "Government officials confirmed on Thursday that the 2024 Olympics were cancelled, according to a statement from the sports ministry.",
]

FAKE = [
    "SHOCKING: Doctors are hiding this one simple trick that cures diabetes overnight - Big Pharma doesn't want you to know!!!",
    "BREAKING: NASA admits the moon landing was filmed in a Hollywood studio, leaked documents reveal",
    "5G towers are spreading the virus, insider whistleblower confirms in explosive secret video",
    "You won't believe what this celebrity said about vaccines - they contain microchips to track you!",
    "Government secretly adding chemicals to tap water to control your mind, share before they delete this",
    "Miracle fruit melts belly fat in 3 days, scientists stunned by this discovery",
    "EXPOSED: Bill Gates plans to depopulate the world with new vaccine, leaked emails prove it",
    "Drinking hot water with lemon kills cancer cells instantly, hospitals refuse to tell patients",
    "The election was rigged by millions of dead voters, mainstream media is covering it up",
    "Aliens have landed in Nevada and the military is hiding them, eyewitnesses claim",
    "Breaking: famous actor found dead in hotel room - police silent, family in shock (click to see)",
    "Scientists confirm the earth is actually flat, NASA forced to delete its photos",
    "New law will ban all cash payments starting next month, banks are panicking, share now!",
    "Eating garlic every day makes you immune to all viruses, top doctor reveals the truth they hide",
    "Secret documents prove the government controls the weather using hidden machines",
    "WhatsApp will start charging users from tomorrow unless you forward this message to 10 contacts",
    "UNESCO declares the new 2000 rupee note the best currency in the world, chip inside tracks black money",
    "NASA confirms the earth will go dark for 6 days next month due to a solar storm, stay indoors!",
    "Cow urine cures cancer and COVID, scientists finally admit the truth suppressed for years",
    "Viral video shows a shark swimming on a flooded highway after the cyclone, authorities stunned",
    "Onions in your room absorb the flu virus, doctors are amazed by this ancient remedy",
    "Famous billionaire giving away free laptops to all students, register on this link before it closes",
    "Leaked: the government will freeze all bank accounts above 50,000 from Monday, withdraw now",
    "Drinking alcohol kills the coronavirus in your throat, health experts don't want you to know",
    "Mobile phone radiation is cooking your brain, secret report the telecom companies buried",
    "Scientists reveal that the pyramids were built as power plants by an ancient advanced civilization, mainstream historians silent",
    "Hospitals are paid $40,000 for every patient they label as COVID, nurse whistleblower exposes",
    "Eating a spoon of turmeric every morning reverses heart disease completely, cardiologists furious",
    "Breaking: popular soft drink found to contain HIV-infected blood, share to warn your family",
    "The moon will appear as big as the earth tonight for the first time in 1000 years, don't miss it",
]
