import logging

# Add your own countries/subdivisions if you wish to generalize
REGIONS = {
    # Republics
    "Respublika Adygeya": {
        "aliases": ["Republic of Adygea", "Adygea", "Adygeya", "Adyghea", "Adygheia", "AD", "ADY"]
    },
    "Respublika Altay": {
        "aliases": ["Republic of Altai", "Altai Republic", "Gorniy Altai", "Gorny Altai", "AL", "Gorno"]
    },
    "Bashkortostan Republic": {
        "aliases": ["Republic of Bashkortostan", "Bashkiria", "Bashkortostan", "Bashkir", "BA", "bash", "bashkort", "ufa"]
    },
    "Respublika Buryatiya": {
        "aliases": ["Republic of Buryatia", "Buryatia", "Buryat", "Buryatiya", "BU", "buryat", "Majin Buu", "Buu"]
    },
    "Chechenskaya Respublika": {
        "aliases": ["Chechen Republic", "Chechnya", "Chechenia", "Ichkeria", "Nokhchiyn Respublika", "CE", "chech", "grozny"]
    },
    "Chuvashskaya Respublika": {
        "aliases": ["Chuvash Republic", "Chuvashia", "Chuvash", "CU", "cheboksary"]    
    },
    "Dagestan Republic": {
        "aliases": ["Republic of Dagestan", "Dagestan", "Daghestan", "DA", "Dage", "Dag", "Makhachkala", "Dogestan"]
    },
    "Respublika Ingushetiya": {
        "aliases": ["Republic of Ingushetia", "Ingushetia", "Ingushetiya", "IN", "Ingush", "Ingushetia Republic"]
    },
    "Kabardino-Balkarskaya Respublika": {
        "aliases": ["Kabardino-Balkaria Republic", "Kabardino Balkaria", "Kabardino Balkar", "Kabardin Balkar", "KB", "kab", "Kabardino", "Balkaria", "balkar",
                    "kabardino-balkarian republic", "kabardino-balkaria", "kabardion", "kabardino balkarian republic", "nalchik"]
    },
    "Kalmykiya": {
        "aliases": ["Republic of Kalmykia", "Kalmykia", "Kalmyk", "Kalmykiya", "Khalmg Tangch", "KL", "elista"]
    },
    "Karachayevo-Cherkesiya": {
        "aliases": ["Karachay-Cherkess Republic", "Karachay Cherkessia", "Karachai Cherkess", "Karachayevo Cherkessiya", "KC", "karachay", "cherkess",
                    "cherkessiya", "cherkessia", "karachay-cherkessia", "Cherkessk"]
    },
    "Respublika Kareliya": {
        "aliases": ["Republic of Karelia", "Karelia", "Kareliya", "KR", "rok", "Karjala", "Finland", "Suomi"]
    },
    "Respublika Khakasiya": {
        "aliases": ["Republic of Khakassia", "Khakassia", "Khakassiya", "Khakasia", "KK", "khakassia republic", "abakan"]
    },
    "Komi": {
        "aliases": ["Komi Republic", "Komi", "KO", "Komi-Permyak", "Komi-Permyak Autonomous Okrug", "Komi-Permyak Autonomous District"]
    },
    "Respublika Mariy-El": {
        "aliases": ["Mari El Republic", "Mari El", "Mari", "Mariy El", "ME", "Yoshkar-Ola", "yoshkar", "mari", "el mari"]
    },
    "Respublika Mordoviya": {
        "aliases": ["Republic of Mordovia", "Mordovia", "Mordoviya", "MO", "Saransk"]
    },
    "Respublika Sakha (Yakutiya)": {
        "aliases": ["Republic of Sakha", "Sakha", "Yakutia", "Yakutsk", "Yakutiya", "SA"]
    },
    "North Ossetia Republic": {
        "aliases": ["Republic of North Ossetia-Alania", "North Ossetia–Alania Republic", "Respublika Severnaya Osetia-Alania", "North Ossetia–Alania",
                     "North Ossetia", "Alania", "Ossetia", "Ironston", "Iron", "Alaniya", "SE", "Vladikavkaz"]
    },
    "Tatarstan": {
        "aliases": ["Republic of Tatarstan", "Tatarstan", "Tatar", "Tataria", "TA", "Kazan"]
    },
    "Respublika Tyva": {
        "aliases": ["Republic of Tuva", "Tuva", "Tyva", "Tannu Tuva", "TY"]
    },
    "Udmurtskaya Respublika": {
        "aliases": ["Udmurt Republic", "Udmurtia", "Udmurt", "UD", "Izhevsk"]
    },
    
    # Krais
    "Altayskiy Kray": {
        "aliases": ["Altai Krai", "Altay Krai", "ALT", "Barnaul"]
    },
    "Kamchatka Krai": {
        "aliases": ["Kamchatka Krai", "Kamchatka", "Kamchatsky", "KAM", "Petropavlovsk-Kamchatsky", "Petropavlovsk"]
    },
    "Khabarovskiy Kray": {
        "aliases": ["Khabarovsk Krai", "Khabarovsk", "Khabarovsky", "KHA", "Khab"]
    },
    "Krasnodarskiy Kray": {
        "aliases": ["Krasnodar Krai", "Krasnodar", "Kuban", "KDA"]
    },
    "Krasnoyarskiy Kray": {
        "aliases": ["Krasnoyarsk Krai", "Krasnoyarsk", "Krasnoyarsky", "KYA"]
    },
    "Perm Krai": {
        "aliases": ["Perm Krai", "Perm", "Permsky", "PER"]
    },
    "Primorskiy Kray": {
        "aliases": ["Primorsky Krai", "Primorsky", "Primorye", "PRI", "Vladivostok", "Nakhodka", "PRIM", "Primcess and the Frog"]
    },
    "Stavropol Kray": {
        "aliases": ["Stavropol Krai", "Stavropol", "Stavropolsky", "STA", "Pyatigorsk", "Stav", "Stavropolitan"]
    },
    "Transbaikal Territory": {
        "aliases": ["Zabaykalsky Krai", "Zabaykalsky", "Zabaikalsky", "Transbaikal", "ZAB", "Zabay", "Chita", "Cheetah"]
    },

    # Oblasts
    "Amur Oblast": {
        "aliases": ["Amur Oblast", "Amur", "Amurskaya", "AMU", "Blagoveshchensk", "Blagoveshchenskaya", "Blago"]
    },
    "Arkhangelsk Oblast": {
        "aliases": ["Arkhangelsk Oblast", "Arkhangelsk", "Arkhangelskaya", "ARK"]
    },
    "Astrakhan Oblast": {
        "aliases": ["Astrakhan Oblast", "Astrakhan", "Astrakhanskaya", "AST"]
    },
    "Belgorod Oblast": {
        "aliases": ["Belgorod Oblast", "Belgorod", "Belgorodskaya", "BEL", "Belgo"]
    },
    "Bryansk Oblast": {
        "aliases": ["Bryansk Oblast", "Bryansk", "Bryanskaya", "BRY"]
    },
    "Chelyabinsk Oblast": {
        "aliases": ["Chelyabinsk Oblast", "Chelyabinsk", "Chelyabinskaya", "CHE", "Chelya"]
    },
    "Irkutsk Oblast": {
        "aliases": ["Irkutsk Oblast", "Irkutsk", "Irkutskaya", "IRK", "Bratsk"]
    },
    "Ivanovo Oblast": {
        "aliases": ["Ivanovo Oblast", "Ivanovo", "Ivanovskaya", "IVA"]
    },
    "Kaliningrad Oblast": {
        "aliases": ["Kaliningrad Oblast", "Kaliningrad", "Kaliningradskaya", "KGD", "Poland", "Lithuania", "Teutonic Order", "Koenigsberg",
                    "Karaliaučius", "Królewiec", "Kunnegsgarbs", "Kyonigsberg", "Královec", "Königsberg"]
    },
    "Kaluga Oblast": {
        "aliases": ["Kaluga Oblast", "Kaluga", "Kaluzhskaya", "KLU"]
    },
    "Kemerovo Oblast": {
        "aliases": ["Kemerovo Oblast", "Kemerovo", "Kemerovskaya", "Kuzbass", "KEM"]
    },
    "Kirov Oblast": {
        "aliases": ["Kirov Oblast", "Kirov", "Kirovskaya", "KIR"]
    },
    "Kostroma Oblast": {
        "aliases": ["Kostroma Oblast", "Kostroma", "Kostromskaya", "KOS"]
    },
    "Kurgan Oblast": {
        "aliases": ["Kurgan Oblast", "Kurgan", "Kurganskaya", "KGN"]
    },
    "Kursk Oblast": {
        "aliases": ["Kursk Oblast", "Kursk", "Kurskaya", "KRS", "Ukraine"]
    },
    "Leningrad Oblast": {
        "aliases": ["Leningrad Oblast", "Leningrad", "Leningradskaya", "LEN", "Lenin"]
    },
    "Lipetsk Oblast": {
        "aliases": ["Lipetsk Oblast", "Lipetsk", "Lipetskaya", "LIP"]
    },
    "Magadan Oblast": {
        "aliases": ["Magadan Oblast", "Magadan", "Magadanskaya", "MAG", "maga"]
    },
    "Moscow Oblast": {
        "aliases": ["Moscow Oblast", "Moscow Oblast", "Moskovskaya", "Podmoskovye", "MOS"]
    },
    "Murmansk Oblast": {
        "aliases": ["Murmansk Oblast", "Murmansk", "Murmanskaya", "MUR"]
    },
    "Nizhny Novgorod Oblast": {
        "aliases": ["Nizhny Novgorod Oblast", "Nizhny Novgorod", "Nizhegorodskaya", "Nizhny", "NIZ"]
    },
    "Novgorod Oblast": {
        "aliases": ["Novgorod Oblast", "Novgorod", "Novgorodskaya", "NGR"]
    },
    "Novosibirsk Oblast": {
        "aliases": ["Novosibirsk Oblast", "Novosibirsk", "Novosibirskaya", "NVS", "Novo"]
    },
    "Omsk Oblast": {
        "aliases": ["Omsk Oblast", "Omsk", "Omskaya", "OMS"]
    },
    "Orenburg Oblast": {
        "aliases": ["Orenburg Oblast", "Orenburg", "Orenburgskaya", "ORE", "oren"]
    },
    "Orel Oblast": {
        "aliases": ["Oryol Oblast", "Oryol", "Orlovskaya", "Orel", "ORL", "Oreo"]
    },
    "Penza Oblast": {
        "aliases": ["Penza Oblast", "Penza", "Penzenskaya", "PNZ", "pen", "PEZ"]
    },
    "Pskov Oblast": {
        "aliases": ["Pskov Oblast", "Pskov", "Pskovskaya", "PSK"]
    },
    "Rostov Oblast": {
        "aliases": ["Rostov Oblast", "Rostov", "Rostovskaya", "ROS"]
    },
    "Ryazan Oblast": {
        "aliases": ["Ryazan Oblast", "Ryazan", "Ryazanskaya", "RYA"]
    },
    "Sakhalin Oblast": {
        "aliases": ["Sakhalin Oblast", "Sakhalin", "Sakhalinskaya", "SAK"]
    },
    "Samara Oblast": {
        "aliases": ["Samara Oblast", "Samara", "Samarskaya", "SAM", "Stavropol-on-Volga"]
    },
    "Saratovskaya Oblast": {
        "aliases": ["Saratov Oblast", "Saratov", "Saratovskaya", "SAR"]
    },
    "Smolensk Oblast": {
        "aliases": ["Smolensk Oblast", "Smolensk", "Smolenskaya", "SMO"]
    },
    "Sverdlovsk Oblast": {
        "aliases": ["Sverdlovsk Oblast", "Sverdlovsk", "Sverdlovskaya", "SVE", "Yekaterinburg", "Ekaterinburg", "Yeka", "yek", "Yekat"]
    },
    "Tambov Oblast": {
        "aliases": ["Tambov Oblast", "Tambov", "Tambovskaya", "TAM"]
    },
    "Tomsk Oblast": {
        "aliases": ["Tomsk Oblast", "Tomsk", "Tomskaya", "TOM"]
    },
    "Tula Oblast": {
        "aliases": ["Tula Oblast", "Tula", "Tulskaya", "TUL"]
    },
    "Tver Oblast": {
        "aliases": ["Tver Oblast", "Tver", "Tverskaya", "TVE"]
    },
    "Tyumenskaya Oblast’": {
        "aliases": ["Tyumen Oblast", "Tyumen", "Tyumenskaya", "TYU"]
    },
    "Ulyanovsk Oblast": {
        "aliases": ["Ulyanovsk Oblast", "Ulyanovsk", "Ulyanovskaya", "ULY", "Ulya"]
    },
    "Vladimirskaya Oblast’": {
        "aliases": ["Vladimir Oblast", "Vladimir", "Vladimirskaya", "VLA"]
    },
    "Volgograd Oblast": {
        "aliases": ["Volgograd Oblast", "Volgograd", "Volgogradskaya", "VGG", "Stalingrad", "Volgo"]
    },
    "Vologda Oblast": {
        "aliases": ["Vologda Oblast", "Vologda", "Vologodskaya", "VLG"]
    },
    "Voronezh Oblast": {
        "aliases": ["Voronezh Oblast", "Voronezh", "Voronezhskaya", "VOR"]
    },
    "Yaroslavl Oblast": {
        "aliases": ["Yaroslavl Oblast", "Yaroslavl", "Yaroslavskaya", "YAR", "Yaro"]
    },

    # Autonomous Okrugs
    "Chukotka Autonomous Okrug": {
        "aliases": ["Chukotka Autonomous Okrug", "Chukotka", "Chukotsky", "CHU"]
    },
    "Khanty-Mansiyskiy Avtonomnyy Okrug-Yugra": {
        "aliases": ["Khanty-Mansi Autonomous Okrug", "Khanty Mansi", "Jugra", "Yugra", "Khantia Mansia", "KHM", "Khanty", "Khanty-Mansi", "Khanty-Mansiysk",
                     "Khanty-Mansiyskiy", "Khanty-Mansiyskiy Avtonomnyy Okrug", "Khanty-Mansiyskiy Avtonomnyy Okrug-Yugra", "Khanty Mansi Autonomous Okrug", "Mansi",
                     "Surgut"]
    },
    "Nenetskiy Avtonomnyy Okrug": {
        "aliases": ["Nenets Autonomous Okrug", "Nenets", "Nenetsky", "NEN"]
    },
    "Yamalo-Nenetskiy Avtonomnyy Okrug": {
        "aliases": ["Yamalo-Nenets Autonomous Okrug", "Yamalo Nenets", "Yamal", "YAN", "Yamalo", "Ian Nepomniatchi", "Yanmega"]
    },

    # Federal Cities
    "Moscow Federal City": {
        "aliases": ["Moscow", "Moskva", "MOW"]
    },
    "Saint Petersburg Federal City": {
        "aliases": ["Saint Petersburg", "Saint Petersburg", "St Petersburg", "SPb", "Petersburg", "SPE"]
    },
    "Sevastopol Federal City": {
        "aliases": ["Sevastopol", "Sevastopol", "SEV"]
    },

    # Autonomous Oblast
    "Yevrey (Jewish) Autonomous Oblast": {
        "aliases": ["Jewish Autonomous Oblast", "Jewish", "Yevrey", "Birobidzhan", "YEV", "JAO", "Israel"]
    }
}

class RegionFlatmap:
    """Flat map where each alias points to alias list with standardized name first"""
    def __init__(self, subdivisions_data):
        self.flat_map = {}
        
        for canonical_name, data in subdivisions_data.items():
            # Get standardized name (first alias) and rest of aliases
            standardized_name = data['aliases'][0]  # First alias is standardized name
            other_aliases = data['aliases'][1:] + [canonical_name]  # Rest of aliases + canonical
            
            # Create ordered alias list with standardized name first
            all_aliases = [standardized_name] + other_aliases
            
            # Map each alias to the full list
            for alias in all_aliases:
                self.flat_map[alias.lower()] = all_aliases

    def verify_guess(self, guess: str, actual: str) -> bool:
        """
        Verify if a guess matches any alias of the actual location
        Returns True if guess and actual share any aliases
        """
        if not guess:
            logging.error("Invalid guess")
        if not actual:
            logging.error("Invalid actual")

        guess = guess.lower().strip()
        actual = actual.lower().strip()
        
        if guess not in self.flat_map or actual not in self.flat_map:
            return False
        
        return self.flat_map[guess] == self.flat_map[actual]

    def get_canonical_name(self, location: str) -> str:
        """Get the first alias (canonical) for any valid alias"""
        location = location.lower().strip()
        if location in self.flat_map:
            return self.flat_map[location][0]
        return None

    def is_valid_location(self, location: str) -> bool:
        """Check if a string is a valid location alias"""
        return location.lower().strip() in self.flat_map

    def get_all_aliases(self, location: str) -> list:
        """Get all aliases for a location"""
        location = location.lower().strip()
        if location in self.flat_map:
            return self.flat_map[location]
        return []