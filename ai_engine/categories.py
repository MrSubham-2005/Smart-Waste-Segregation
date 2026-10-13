"""
EcoSort AI - Waste Categories Definition & Rule Engine
Defines the 4 core classification categories, bin color mappings,
disposal guidelines, and environmental impact context.
"""

WASTE_CATEGORIES = {
    0: {
        'id': 1,
        'index': 0,
        'name': 'Wet Waste',
        'key': 'wet_waste',
        'bin_name': 'Green Bin (Organic / Compostable)',
        'bin_color_code': '#28a745',
        'badge_class': 'badge-wet',
        'description': (
            'Biodegradable organic waste from plants or animals that decomposes '
            'naturally through aerobic or anaerobic microbial action.'
        ),
        'disposal_instructions': (
            'Dispose directly into the green bin without plastic or polythene liners. '
            'Ideal for household composters, community biogas, or municipal biomethanation processing.'
        ),
        'examples': (
            'Fruit and vegetable peels, leftover cooked food, tea bags, coffee grounds, '
            'eggshells, garden trimmings, dried leaves, fallen flowers.'
        ),
        'sdg_impact': (
            'Diverting organic waste from open landfills prevents uncontrolled methane generation '
            'and substantially reduces waste generation through composting and recycling.'
        )
    },
    1: {
        'id': 2,
        'index': 1,
        'name': 'Dry Waste',
        'key': 'dry_waste',
        'bin_name': 'Blue Bin (Non-Recyclable Dry)',
        'bin_color_code': '#007bff',
        'badge_class': 'badge-dry',
        'description': (
            'Non-biodegradable, non-recyclable solid waste that cannot be composted '
            'or mechanically recovered into secondary raw materials.'
        ),
        'disposal_instructions': (
            'Ensure items are reasonably dry before placing in the blue bin to prevent odor '
            'and microbial proliferation. Keep strictly separated from wet kitchen items.'
        ),
        'examples': (
            'Soiled paper wrappers, non-recyclable multi-layered packaging (MLP), chip packets, '
            'broken ceramics, rubber scraps, sweeping dust, sanitary products.'
        ),
        'sdg_impact': (
            'Prevents cross-contamination of clean recyclable streams and optimizes waste-to-energy '
            'and sanitary landfill logistics.'
        )
    },
    2: {
        'id': 3,
        'index': 2,
        'name': 'Recyclable Waste',
        'key': 'recyclable_waste',
        'bin_name': 'Yellow Bin (Clean Recyclables)',
        'bin_color_code': '#ffc107',
        'badge_class': 'badge-recyclable',
        'description': (
            'Clean, non-contaminated materials that can be mechanically or chemically '
            'reprocessed into new consumer or industrial products.'
        ),
        'disposal_instructions': (
            'Rinse containers clean of food residue and air-dry thoroughly before disposal. '
            'Flatten cardboard cartons and compress plastic bottles to maximize collection capacity.'
        ),
        'examples': (
            'Clear PET plastic bottles, HDPE milk containers, aluminum beverage cans, '
            'tin cans, cardboard boxes, clean newspapers, magazines, glass bottles.'
        ),
        'sdg_impact': (
            'Reduces raw virgin resource extraction and industrial greenhouse gas emissions '
            'by supplying circular raw materials for manufacturing.'
        )
    },
    3: {
        'id': 4,
        'index': 3,
        'name': 'E-Waste',
        'key': 'e_waste',
        'bin_name': 'Red Bin (Specialized E-Waste Bin)',
        'bin_color_code': '#dc3545',
        'badge_class': 'badge-ewaste',
        'description': (
            'Discarded electrical and electronic equipment (WEEE) containing hazardous heavy metals '
            '(lead, mercury, cadmium) as well as recoverable precious metals (gold, copper, rare earths).'
        ),
        'disposal_instructions': (
            'DO NOT discard with regular household garbage or landfill bins. Store in a dry indoor '
            'container and hand over to authorized government e-waste collection drives or certified recyclers.'
        ),
        'examples': (
            'Old mobile phones, chargers, USB cords, earphones, remote controls, '
            'printed circuit boards (PCBs), consumer batteries, computer accessories.'
        ),
        'sdg_impact': (
            'Environmentally sound management of hazardous electronic chemicals throughout their '
            'life cycle to minimize adverse health and water contamination impacts.'
        )
    }
}

CATEGORY_NAMES = [WASTE_CATEGORIES[i]['name'] for i in range(len(WASTE_CATEGORIES))]


def get_category_by_index(index: int) -> dict:
    """Retrieve full category metadata by model softmax index (0..3)."""
    return WASTE_CATEGORIES.get(index)


def get_category_by_name(name: str) -> dict:
    """Retrieve category metadata by human-readable category name."""
    clean_name = name.strip().lower()
    for cat in WASTE_CATEGORIES.values():
        if cat['name'].lower() == clean_name or cat['key'].lower() == clean_name:
            return cat
    return None


def get_category_id_by_name(name: str) -> int:
    """Returns database category ID (1..4) matching the category name."""
    cat = get_category_by_name(name)
    return cat['id'] if cat else None


def get_all_categories() -> list:
    """Returns list of all category definitions sorted by index."""
    return [WASTE_CATEGORIES[i] for i in sorted(WASTE_CATEGORIES.keys())]
