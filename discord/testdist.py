#!/usr/bin/env python
import argparse
import html
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import toml
from ongcodebot import (OngCode, OngCodeIndex, OngCodeMeta, initialize_db,
                        load_title_cache, log)
from peewee import SqliteDatabase
from playhouse.sqlite_ext import SqliteExtDatabase
from rapidfuzz import distance, fuzz, process, utils


def search_titles_with_scorer(
    query_str: str,
    title_cache: List[tuple[int, str]],
    scorer,
    scorer_name: str,
    limit: int = 10,
    processor: Optional[Callable] = None
) -> List[tuple[int, str, float]]:
    """Search titles using a specific scorer and return results with scores."""
    if not title_cache:
        return []

    # Extract just the titles for fuzzy matching
    titles = [title for _, title in title_cache]

    # Use rapidfuzz to find best matches with the specified scorer
    matches = process.extract(
        query_str,
        titles,
        scorer=scorer,
        limit=limit,
        processor=processor
    )

    # Convert back to (mainmsg_id, title, score) format
    results = []
    for matched_title, score, idx in matches:
        mainmsg_id = title_cache[idx][0]
        results.append((mainmsg_id, matched_title, score))

    return results


def token_aware_scorer(query: str, choice: str, **kwargs) -> float:
    """Enhanced custom scorer combining multiple strategies for optimal matching.

    This scorer is designed to work better than partial_token_sort_ratio by being
    smarter about token boundaries and length mismatches.

    Since this is a simple wrapper around fuzz functions, just use partial_token_sort_ratio
    as the base and enhance it with token-level intelligence.
    """
    # Use partial_token_sort_ratio as our base - it's already very good
    base_score = fuzz.partial_token_sort_ratio(query, choice, processor=utils.default_process)

    # Apply processor ourselves to do token analysis
    query_processed = utils.default_process(query)
    choice_processed = utils.default_process(choice)

    # Split choice into tokens
    tokens = choice_processed.split()

    # Find best token match
    best_token_score = 0
    for token in tokens:
        # Direct token matching with partial_ratio
        token_score = fuzz.ratio(query_processed, token)

        # Strong bonus for prefix matches
        if len(query_processed) >= 3 and token.startswith(query_processed):
            token_score = max(token_score, 92)

        # Penalize length mismatches for short queries
        if len(query_processed) <= 5 and len(token) > len(query_processed) * 2.5:
            token_score *= 0.6

        best_token_score = max(best_token_score, token_score)

    # Return the better of base score or token score
    return max(base_score, best_token_score)


def normalize_for_comparison(text: str) -> str:
    """Normalize text by removing diacritical marks for comparison purposes.

    This ensures that 'Queensrÿche' and 'Queensryche' are considered identical.
    """
    # NFD = Canonical Decomposition (separates base characters from diacritics)
    # Then filter out combining characters (diacritical marks)
    nfd = unicodedata.normalize('NFD', text)
    without_diacritics = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    return without_diacritics.lower()


def get_test_dataset():
    """Return a curated test dataset of titles and realistic search queries.

    Returns:
        List of tuples: (expected_title, [list of search queries])
    """
    return [
        # Metal bands - common misspellings and variations
        ("Queensrÿche", [
            "quensrich", "queensryche", "queensrych", "queesryche", "queensreich",
            "queenryche", "qeensryche", "queens ryche", "qryche", "queensrike",
            "queensrich", "quensryche", "queenryke", "queensriche", "qeensrich",
            "queesrich", "queenreich", "queen ryche", "queensrÿke", "queen's ryche"
        ]),
        ("Metallica", [
            "metalica", "mettalica", "metallika", "metalllica", "metlica",
            "metallca", "metalliac", "metalic", "mettallica", "metalllika",
            "metalika", "mettalika", "metllica", "metaliica", "metal lica",
            "metalika", "metalicka", "m3tallica", "metallika"
        ]),
        ("Iron Maiden", [
            "iron maden", "ironmaiden", "iron maidan", "iorn maiden", "iron madin",
            "iron maidon", "maiden", "iron mayden", "iron maideon", "irom maiden",
            "iron maided", "ironmaidon", "iron maide", "irn maiden", "iron maien",
            "iron madan", "maiden iron", "i maiden", "iron maydn"
        ]),
        ("Judas Priest", [
            "judas preist", "judaspriest", "judus priest", "judas preest", "judas preast",
            "judis priest", "judas prist", "judus preist", "judis preist", "jdas priest",
            "judus preast", "judas priist", "judas prieest", "judah priest", "priest judas",
            "judas pr1est", "judas preest"
        ]),
        ("Black Sabbath", [
            "black sabath", "blacksabbath", "blak sabbath", "black sabith", "black sabbith",
            "sabbath", "black sab", "blck sabbath", "black sabbeth", "blck sabath",
            "black sababth", "blak sabath", "black sabbbath", "black sabeth", "balck sabbath",
            "blk sabbath", "black sabbth", "sabbath black", "b sabbath"
        ]),
        ("Slayer", [
            "slyer", "slayr", "slayar", "slaer", "slayyer",
            "slaayer", "slaeyr", "sleyer", "slaier", "slayeer",
            "slayerr", "slaaer", "sl4yer", "slayer band", "slay3r"
        ]),
        ("Megadeth", [
            "megadeath", "megadth", "megadeh", "megaeth", "megedeth",
            "megadith", "mega death", "megadth", "megadead", "meggadeth",
            "megadeff", "megadth", "m3gadeth", "mega deth", "megadth"
        ]),
        ("Pantera", [
            "pantera", "pantra", "pntera", "pantera band", "pantera metal",
            "pantara", "pantera group", "panter", "pantera tx", "panteraa"
        ]),
        ("Anthrax", [
            "anthax", "anthraks", "anthracks", "anthrx", "antrax",
            "anthraxx", "anthraks", "anthracs", "anthrax band", "anthraks"
        ]),
        ("Dream Theater", [
            "dreamtheater", "dream theatre", "dream theator", "dreem theater", "drem theater",
            "dream theatr", "dreamtheatre", "dream theatur", "dream theter", "dreem theatre",
            "dream theater band", "theater dream", "dream thtr", "dream theater"
        ]),
        # Song titles with complex words
        ("Master of Puppets", [
            "master of pupets", "masterofpuppets", "master puppets", "master of pupits",
            "master of puppits", "puppets", "master puppet", "mastr of puppets",
            "master of pupppets", "master of pupts", "master puppits", "master of pupets",
            "puppets master", "master of puppets song", "master of puppits", "mop",
            "master 0f puppets", "masters of puppets", "master of puppet"
        ]),
        ("Stairway to Heaven", [
            "stairway heaven", "stairwaytoheaven", "stairway to heven", "stairway to heavon",
            "stairwy to heaven", "stairway heavn", "stairway", "heaven", "stairway 2 heaven",
            "stairway to heavan", "stairway too heaven", "stairway to hevn", "stairway to heavin",
            "stairway to heavn", "stairway to heave", "stairway to hevn", "stairway 2 hvn",
            "stairway to heaven song", "stairs to heaven"
        ]),
        ("Bohemian Rhapsody", [
            "bohemian rapsody", "bohemianrhapsody", "bohmian rhapsody", "bohemian rhapsodie",
            "boheamian rhapsody", "rhapsody", "bohemian", "bohemien rhapsody", "bohemian rhappsody",
            "bohmian rapsody", "bohemian rapsody", "bohemian rhapsody song", "bohemian rhapsdy",
            "bohemien rapsody", "bohemian rhapsodi", "bohemian rapsody", "bo rhap"
        ]),
        ("Enter Sandman", [
            "entersandman", "enter sandmann", "enter sandmam", "enter sandmen", "enter sand man",
            "sandman", "ener sandman", "entr sandman", "enter sanman", "enter sandman song",
            "sandman enter", "enter the sandman", "enter sandmann", "enter sndman", "sandmann"
        ]),
        ("Paranoid Android", [
            "paranoidandroid", "paranoid andriod", "paranoid andoid", "paronoid android",
            "paranoid androud", "paranoid androyd", "android", "paranoid", "paranoid androd",
            "paronoid andriod", "paranoid andriod", "paranoid androyd", "paranoid android song",
            "android paranoid", "paranoid andr0id", "paranoid andorid"
        ]),
        ("Smells Like Teen Spirit", [
            "smells like teen sprit", "smells like teenspirit", "smels like teen spirit",
            "smells like teen spirrit", "teen spirit", "smells like", "smells like teen spirt",
            "smells lke teen spirit", "smells like teen spirrit", "smells like teen spirit song",
            "teen spirit smells", "smells like teen spirit nirvana", "smells like teenspirit",
            "smells like spirit", "teen sprit", "smells like teen sperit"
        ]),
        ("One", [
            "one", "oen", "onne", "oone", "one song",
            "one metallica", "1", "one by metallica"
        ]),
        ("The Trooper", [
            "trooper", "the troper", "thetrooper", "the trooper song", "trooper song",
            "the trooper iron maiden", "the trooper maiden", "troopr", "trooper the",
            "trooper maiden", "the trooper im", "troper"
        ]),
        ("Ace of Spades", [
            "ace spades", "ace of spades", "aceofspades", "ace of spades song",
            "ace of spades motorhead", "ace spades song", "aces of spades", "ace of spadez",
            "ace spade", "ace of spds", "ace of spades lemmy"
        ]),
        ("Crazy Train", [
            "crazy train", "crazytrain", "crazy train ozzy", "crazy train song",
            "crazie train", "crazy train osbourne", "crzy train", "crazy trn",
            "crazy train oz", "crazzy train"
        ]),
        ("Breaking the Law", [
            "breakin the law", "breaking law", "breking the law", "braking the law",
            "breaking da law", "breaking te law", "breakingthelaw", "breaking teh law",
            "breaking the lw", "breacking the law", "breaking the lawe", "breakng the law",
            "breaking th law", "breaking the laww", "breaking the law judas", "btl",
            "law breaking", "breaking the law song"
        ]),
        ("Painkiller", [
            "pain killer", "painkilr", "painkiler", "painkillr",
            "painkiller judas", "painkiller song", "pain kilr", "painkillr",
            "painkiller priest", "paynkiller"
        ]),
        ("Holy Diver", [
            "holydiver", "holy divr", "holey diver", "holly diver",
            "holy diver dio", "holy diver song", "holy diver ronnie", "holy divr",
            "holey diver", "holy diver band"
        ]),
        ("Symphony of Destruction", [
            "symphonyofdestruction", "symphony destruction", "symphonie of destruction",
            "symphony of destructon", "symphony of destrution", "symphony of distruction",
            "symphony of destructio", "symphony of destructoin", "symphony of destruction megadeth",
            "symphony of destruction song", "symphony destruction", "symphony of destruct",
            "symphony of destruction", "symphony of destr", "sod"
        ]),
        ("Walk", [
            "walk", "walk pantera", "walk song", "pantera walk", "walk band"
        ]),
        ("Raining Blood", [
            "rainingblood", "raining blod", "rainning blood", "raining blood slayer",
            "raining blood song", "raining blood", "raining bloo", "raining blood slayer song",
            "rainin blood", "raining blud"
        ]),
        ("The Number of the Beast", [
            "number of the beast", "thenumberofthebeast", "the number of beast",
            "number of beast", "the number of the beast iron maiden", "notb",
            "the number of the beast song", "666 number of the beast", "number of the beast maiden",
            "the number of da beast", "the number of the beast", "number beast",
            "the number of the best", "the number of teh beast"
        ]),
        ("Hallowed Be Thy Name", [
            "hallowedbethyname", "hallowed be thy name", "hallowed by thy name",
            "hallowed be thy name maiden", "hallowed be thy name iron maiden", "hallow be thy name",
            "hallowed b thy name", "hallowed be thy name song", "hallowed be thy",
            "hallowed be thy nam", "halowed be thy name"
        ]),
        ("Rainbow in the Dark", [
            "rainbowinthedark", "rainbow in dark", "rainbow in the dark dio",
            "rainbow in the dark song", "rainbow in the dark ronnie", "rainbow n the dark",
            "rainbow in the dark", "rainbow in da dark", "rainbow in the drk",
            "rainbow in the dark band"
        ]),
        ("Peace Sells", [
            "peacesells", "peace sells megadeth", "peace sells song",
            "peace sells but who's buying", "peace sells but", "peace sells...",
            "peace sellz", "peace sells band", "peace sells album"
        ])
    ]


def benchmark_scorers(title_cache: List[tuple[int, str]], limit: int = 10):
    """Comprehensive benchmark of all scorers using realistic test data."""

    # Get test dataset
    test_dataset = get_test_dataset()

    # Fuzz module scorers
    fuzz_scorers = [
        (fuzz.ratio, "fuzz.ratio", "Simple ratio comparison"),
        (fuzz.partial_ratio, "fuzz.partial_ratio", "Best partial string match"),
        (fuzz.token_sort_ratio, "fuzz.token_sort_ratio", "Ratio with sorted tokens"),
        (fuzz.token_set_ratio, "fuzz.token_set_ratio", "Ratio with token sets"),
        (fuzz.token_ratio, "fuzz.token_ratio", "Best of token_sort and token_set"),
        (fuzz.partial_token_sort_ratio, "fuzz.partial_token_sort_ratio", "Partial match with sorted tokens"),
        (fuzz.partial_token_set_ratio, "fuzz.partial_token_set_ratio", "Partial match with token sets"),
        (fuzz.partial_token_ratio, "fuzz.partial_token_ratio", "Best partial token match"),
        (fuzz.WRatio, "fuzz.WRatio", "Weighted ratio (auto-selects best method)"),
        (fuzz.QRatio, "fuzz.QRatio", "Quick ratio (faster, less accurate)"),
    ]

    # Custom scorer
    custom_scorers = [
        (token_aware_scorer, "custom.token_aware", "Token-aware scorer with length penalties for better accuracy"),
    ]

    # Distance module scorers
    distance_scorers = [
        (distance.Levenshtein.normalized_similarity, "distance.Levenshtein", "Edit distance (insertions, deletions, substitutions)"),
        (distance.Indel.normalized_similarity, "distance.Indel", "Edit distance (only insertions and deletions)"),
        (distance.Hamming.normalized_similarity, "distance.Hamming", "Hamming distance (same length strings, substitutions only)"),
        (distance.Jaro.similarity, "distance.Jaro", "Jaro similarity (transpositions)"),
        (distance.JaroWinkler.similarity, "distance.JaroWinkler", "Jaro-Winkler similarity (prefix boost)"),
        (distance.LCSseq.normalized_similarity, "distance.LCSseq", "Longest Common Subsequence"),
        (distance.OSA.normalized_similarity, "distance.OSA", "Optimal String Alignment (Damerau-Levenshtein variant)"),
        (distance.DamerauLevenshtein.normalized_similarity, "distance.DamerauLevenshtein", "Damerau-Levenshtein distance (transpositions)"),
        (distance.Postfix.normalized_similarity, "distance.Postfix", "Postfix similarity"),
        (distance.Prefix.normalized_similarity, "distance.Prefix", "Prefix similarity"),
    ]

    all_scorers = custom_scorers + fuzz_scorers + distance_scorers

    # Always use preprocessing (lowercase, remove non-alphanumeric)
    processor = utils.default_process
    processor_desc = "With preprocessing (default_process: lowercase, remove non-alphanumeric)"

    print("\n🎯 COMPREHENSIVE BENCHMARK MODE")
    print(f"Testing {len(test_dataset)} titles with {sum(len(queries) for _, queries in test_dataset)} total search queries")
    print(f"Title cache size: {len(title_cache)} entries")
    print(f"Testing {len(all_scorers)} scorers with preprocessing")
    print(f"Scorers to test: {', '.join([name for _, name, _ in all_scorers])}")
    print("=" * 120)

    # Build title lookup for quick matching
    title_lookup = {title.lower(): title for _, title in title_cache}

    # Track results for each scorer configuration
    scorer_results = {}

    for _scorer, name, description in all_scorers:
        config_key = name
        scorer_results[config_key] = {
            'name': name,
            'processor': processor_desc,
            'description': description,
            'total_tests': 0,
            'correct_top': 0,
            'correct_in_top_5': 0,
            'total_score_gap': 0.0,  # Sum of (top_score - second_score) for correct results
            'total_time_ms': 0.0,
            'failures': 0
        }

    # Run all tests
    print("\nRunning benchmark tests...")
    for expected_title, search_queries in test_dataset:
        expected_normalized = normalize_for_comparison(expected_title)

        for query in search_queries:
            for scorer, name, _description in all_scorers:
                config_key = name

                try:
                    start_time = time.time()
                    results = search_titles_with_scorer(query, title_cache, scorer, name, limit=5, processor=processor)
                    search_time = (time.time() - start_time) * 1000

                    scorer_results[config_key]['total_tests'] += 1
                    scorer_results[config_key]['total_time_ms'] += search_time

                    if results:
                        top_result_normalized = normalize_for_comparison(results[0][1])

                        # Check if correct result is in top position (ignoring diacritics)
                        if expected_normalized in top_result_normalized or top_result_normalized in expected_normalized:
                            scorer_results[config_key]['correct_top'] += 1

                            # Calculate score gap (margin between top and second result)
                            if len(results) > 1:
                                score_gap = results[0][2] - results[1][2]
                                scorer_results[config_key]['total_score_gap'] += score_gap
                            else:
                                # Only one result, perfect score gap
                                scorer_results[config_key]['total_score_gap'] += results[0][2]

                        # Check if correct result is in top 5 (ignoring diacritics)
                        for _idx, result_title, _score in results[:5]:
                            result_normalized = normalize_for_comparison(result_title)
                            if expected_normalized in result_normalized or result_normalized in expected_normalized:
                                scorer_results[config_key]['correct_in_top_5'] += 1
                                break

                except Exception as e:
                    scorer_results[config_key]['failures'] += 1
                    # Print errors for custom scorers to help debug
                    if 'custom' in name.lower():
                        print(f"Error in {name} with query '{query}': {e}")
                    continue

    # Calculate final metrics and filter to successful configs
    successful_configs = []

    for config_key, results in scorer_results.items():
        if results['total_tests'] == 0:
            continue

        accuracy_top = (results['correct_top'] / results['total_tests']) * 100
        accuracy_top5 = (results['correct_in_top_5'] / results['total_tests']) * 100
        avg_time = results['total_time_ms'] / results['total_tests']
        avg_score_gap = results['total_score_gap'] / max(results['correct_top'], 1)

        # Only include configs with at least 50% accuracy in top result
        if accuracy_top >= 50.0:
            successful_configs.append({
                'config_key': config_key,
                'name': results['name'],
                'processor': results['processor'],
                'description': results['description'],
                'accuracy_top': accuracy_top,
                'accuracy_top5': accuracy_top5,
                'avg_score_gap': avg_score_gap,
                'avg_time_ms': avg_time,
                'correct_top': results['correct_top'],
                'correct_top5': results['correct_in_top_5'],
                'total_tests': results['total_tests'],
                'failures': results['failures']
            })

    # Display all scorer results for debugging (especially custom scorers)
    print("\n📋 ALL SCORER RESULTS:")
    print("=" * 120)
    for config_key, results in sorted(scorer_results.items(), key=lambda x: x[0]):
        if results['total_tests'] > 0:
            accuracy_top = (results['correct_top'] / results['total_tests']) * 100
            print(f"  {config_key}: {accuracy_top:.1f}% accuracy ({results['correct_top']}/{results['total_tests']}), failures: {results['failures']}")

    # Display results
    if not successful_configs:
        print("\n❌ No scorers achieved at least 50% accuracy!")
        return

    print(f"\n✅ Found {len(successful_configs)} scorer configurations with ≥50% accuracy:")
    print("=" * 120)

    # Sort by accuracy (desc), then by score gap (desc), then by time (asc)
    successful_configs.sort(key=lambda x: (-x['accuracy_top'], -x['avg_score_gap'], x['avg_time_ms']))

    # Display top configurations
    print("\n🏆 TOP PERFORMING CONFIGURATIONS:")
    print("=" * 120)

    for i, config in enumerate(successful_configs[:15], 1):  # Show top 15
        print(f"\n{i}. {config['config_key']}")
        print(f"   Description: {config['description']}")
        print(f"   ✓ Top-1 Accuracy: {config['accuracy_top']:.1f}% ({config['correct_top']}/{config['total_tests']})")
        print(f"   ✓ Top-5 Accuracy: {config['accuracy_top5']:.1f}% ({config['correct_top5']}/{config['total_tests']})")
        print(f"   ✓ Avg Score Gap: {config['avg_score_gap']:.2f} (higher is better)")
        print(f"   ✓ Avg Search Time: {config['avg_time_ms']:.2f}ms")
        if config['failures'] > 0:
            print(f"   ⚠ Failures: {config['failures']}")

    # Show summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY:")
    best = successful_configs[0]
    print(f"  🥇 Best Overall: {best['config_key']}")
    print(f"     - Top-1 Accuracy: {best['accuracy_top']:.1f}%")
    print(f"     - Avg Score Gap: {best['avg_score_gap']:.2f}")
    print(f"     - Avg Time: {best['avg_time_ms']:.2f}ms")

    fastest = min(successful_configs, key=lambda x: x['avg_time_ms'])
    print(f"  ⚡ Fastest: {fastest['config_key']} ({fastest['avg_time_ms']:.2f}ms, {fastest['accuracy_top']:.1f}% accuracy)")

    best_gap = max(successful_configs, key=lambda x: x['avg_score_gap'])
    print(f"  📊 Best Score Gap: {best_gap['config_key']} ({best_gap['avg_score_gap']:.2f}, {best_gap['accuracy_top']:.1f}% accuracy)")

    print(f"\n  Total configurations tested: {len(all_scorers)}")
    print(f"  Configurations with ≥50% accuracy: {len(successful_configs)}")

    # Generate HTML report
    generate_html_report(test_dataset, title_cache, best, successful_configs, all_scorers)


def generate_html_report(
    test_dataset: List[tuple[str, List[str]]],
    title_cache: List[tuple[int, str]],
    best_config: dict,
    successful_configs: List[dict],
    all_scorers: List[tuple]
):
    """Generate an HTML report showing benchmark results and detailed query breakdowns."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_filename = Path(__file__).parent / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    # Collect detailed results for ALL successful configurations
    processor = utils.default_process
    all_detailed_results = {}  # scorer_name -> list of results

    print("\n📊 Generating detailed HTML report for all successful configurations...")

    for config in successful_configs:
        scorer_name = config['config_key']

        # Find the scorer function
        scorer_func = None
        for scorer, name, _desc in all_scorers:
            if name == scorer_name:
                scorer_func = scorer
                break

        if scorer_func is None:
            print(f"  ⚠️  Could not find scorer {scorer_name}, skipping...")
            continue

        print(f"  Processing {scorer_name}...")
        detailed_results = []

        for expected_title, search_queries in test_dataset:
            for query in search_queries:
                results = search_titles_with_scorer(query, title_cache, scorer_func, scorer_name, limit=5, processor=processor)

                # Check if the top result is correct (ignoring diacritics)
                expected_normalized = normalize_for_comparison(expected_title)
                is_correct = False
                correct_rank = None

                for rank, (_idx, result_title, _score) in enumerate(results, 1):
                    result_normalized = normalize_for_comparison(result_title)
                    if expected_normalized in result_normalized or result_normalized in expected_normalized:
                        if rank == 1:
                            is_correct = True
                        correct_rank = rank
                        break

                detailed_results.append({
                    'expected': expected_title,
                    'query': query,
                    'results': results,
                    'is_correct': is_correct,
                    'correct_rank': correct_rank
                })

        all_detailed_results[scorer_name] = detailed_results

    # Build HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fuzzy Search Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: white;
        }}
        .header .timestamp {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .summary {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #667eea;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .top-configs {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .config-card {{
            background: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #764ba2;
        }}
        .config-name {{
            font-weight: bold;
            font-size: 18px;
            color: #333;
            margin-bottom: 5px;
        }}
        .config-desc {{
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .config-stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .config-stat {{
            font-size: 13px;
            color: #555;
        }}
        .config-stat strong {{
            color: #667eea;
        }}
        .detailed-results {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .query-block {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #ddd;
        }}
        .query-block.correct {{
            border-left-color: #28a745;
            background: #f1f9f3;
        }}
        .query-block.incorrect {{
            border-left-color: #dc3545;
            background: #fdf3f4;
        }}
        .query-header {{
            margin-bottom: 15px;
        }}
        .expected-title {{
            font-weight: bold;
            color: #333;
            font-size: 16px;
        }}
        .search-query {{
            color: #666;
            font-family: 'Courier New', monospace;
            background: white;
            padding: 5px 10px;
            border-radius: 4px;
            display: inline-block;
            margin-top: 5px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .status-badge.correct {{
            background: #28a745;
            color: white;
        }}
        .status-badge.incorrect {{
            background: #dc3545;
            color: white;
        }}
        .status-badge.partial {{
            background: #ffc107;
            color: #333;
        }}
        .results-list {{
            list-style: none;
            padding: 0;
            margin: 10px 0 0 0;
        }}
        .result-item {{
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .result-item.rank-1 {{
            border: 2px solid #667eea;
            font-weight: bold;
        }}
        .result-rank {{
            color: #667eea;
            font-weight: bold;
            margin-right: 10px;
            min-width: 30px;
        }}
        .result-title {{
            flex-grow: 1;
            color: #333;
        }}
        .result-score {{
            color: #764ba2;
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }}
        .filter-controls {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 20px 0;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0;
        }}
        .tab-button {{
            background: #f8f9fa;
            border: none;
            border-bottom: 3px solid transparent;
            padding: 12px 24px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            transition: all 0.3s ease;
            border-radius: 5px 5px 0 0;
        }}
        .tab-button:hover {{
            background: #e9ecef;
            color: #333;
        }}
        .tab-button.active {{
            background: white;
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
        }}
        .filter-controls label {{
            margin-right: 20px;
            cursor: pointer;
        }}
        .filter-controls input[type="radio"] {{
            margin-right: 5px;
        }}
        .stats-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Fuzzy Search Benchmark Report</h1>
        <div class="timestamp">Generated: {html.escape(timestamp)}</div>
    </div>

    <div class="summary">
        <h2>🏆 Best Configuration</h2>
        <div class="stats-summary">
            <div class="metric">
                <div class="metric-label">Algorithm</div>
                <div class="metric-value" style="font-size: 18px;">{html.escape(best_config['config_key'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Top-1 Accuracy</div>
                <div class="metric-value">{best_config['accuracy_top']:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Top-5 Accuracy</div>
                <div class="metric-value">{best_config['accuracy_top5']:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Score Gap</div>
                <div class="metric-value">{best_config['avg_score_gap']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Search Time</div>
                <div class="metric-value">{best_config['avg_time_ms']:.2f}ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Tests</div>
                <div class="metric-value">{best_config['total_tests']}</div>
            </div>
        </div>
        <p><strong>Description:</strong> {html.escape(best_config['description'])}</p>
    </div>

    <div class="top-configs">
        <h2>📊 Top Performing Configurations</h2>
"""

    for i, config in enumerate(successful_configs[:10], 1):
        html_content += f"""
        <div class="config-card">
            <div class="config-name">#{i} {html.escape(config['config_key'])}</div>
            <div class="config-desc">{html.escape(config['description'])}</div>
            <div class="config-stats">
                <div class="config-stat"><strong>Top-1:</strong> {config['accuracy_top']:.1f}%</div>
                <div class="config-stat"><strong>Top-5:</strong> {config['accuracy_top5']:.1f}%</div>
                <div class="config-stat"><strong>Score Gap:</strong> {config['avg_score_gap']:.2f}</div>
                <div class="config-stat"><strong>Time:</strong> {config['avg_time_ms']:.2f}ms</div>
            </div>
        </div>
"""

    html_content += """
    </div>

    <div class="detailed-results">
        <h2>🔍 Detailed Query Results</h2>
        <p>Showing top 5 results for each query across all successful configurations (≥50% accuracy).</p>

        <div class="tabs">
"""

    # Add tabs for each scorer
    first = True
    for scorer_name in all_detailed_results.keys():
        active_class = "active" if first else ""
        html_content += f"""            <button class="tab-button {active_class}" onclick="openTab(event, '{scorer_name}')">{html.escape(scorer_name)}</button>\n"""
        first = False

    html_content += """        </div>

        <div class="filter-controls">
            <strong>Filter:</strong>
            <label><input type="radio" name="filter" value="all" checked onchange="filterResults(this.value)"> All Results</label>
            <label><input type="radio" name="filter" value="correct" onchange="filterResults(this.value)"> ✅ Correct Only</label>
            <label><input type="radio" name="filter" value="incorrect" onchange="filterResults(this.value)"> ❌ Incorrect Only</label>
        </div>
"""

    # Add detailed results for each scorer
    first = True
    for scorer_name, detailed_results in all_detailed_results.items():
        display_style = "block" if first else "none"
        html_content += f"""
        <div id="{scorer_name}" class="tab-content" style="display: {display_style};">
"""

        for detail in detailed_results:
            status_class = "correct" if detail['is_correct'] else "incorrect"

            if detail['is_correct']:
                status_badge = '<span class="status-badge correct">✅ CORRECT</span>'
            elif detail['correct_rank']:
                status_badge = f'<span class="status-badge partial">⚠️ Rank #{detail["correct_rank"]}</span>'
            else:
                status_badge = '<span class="status-badge incorrect">❌ NOT IN TOP 5</span>'

            html_content += f"""
            <div class="query-block {status_class}" data-status="{status_class}">
                <div class="query-header">
                    <div class="expected-title">Expected: {html.escape(detail['expected'])} {status_badge}</div>
                    <div class="search-query">Query: "{html.escape(detail['query'])}"</div>
                </div>
                <ul class="results-list">
"""

            for rank, (_idx, title, score) in enumerate(detail['results'], 1):
                rank_class = "rank-1" if rank == 1 else ""
                html_content += f"""
                    <li class="result-item {rank_class}">
                        <span class="result-rank">#{rank}</span>
                        <span class="result-title">{html.escape(title)}</span>
                        <span class="result-score">{score:.2f}</span>
                    </li>
"""

            html_content += """
                </ul>
            </div>
"""

        html_content += """
        </div>
"""
        first = False

    html_content += """
    </div>

    <script>
        function openTab(evt, tabName) {
            // Hide all tab contents
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].style.display = 'none';
            }

            // Remove active class from all tab buttons
            const tabButtons = document.getElementsByClassName('tab-button');
            for (let i = 0; i < tabButtons.length; i++) {
                tabButtons[i].className = tabButtons[i].className.replace(' active', '');
            }

            // Show the selected tab and mark button as active
            document.getElementById(tabName).style.display = 'block';
            evt.currentTarget.className += ' active';
        }

        function filterResults(filter) {
            const blocks = document.querySelectorAll('.query-block');
            blocks.forEach(block => {
                if (filter === 'all') {
                    block.style.display = 'block';
                } else {
                    block.style.display = block.dataset.status === filter ? 'block' : 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

    # Write the report
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Calculate stats for the best scorer
    best_scorer_results = all_detailed_results[best_config['config_key']]

    print(f"\n✅ HTML report generated: {report_filename}")
    print(f"   Configurations included: {len(all_detailed_results)}")
    print(f"   Total queries analyzed (per config): {len(best_scorer_results)}")
    print(f"   Best scorer ({best_config['config_key']}) correct: {sum(1 for d in best_scorer_results if d['is_correct'])}")
    print(f"   Best scorer ({best_config['config_key']}) incorrect: {sum(1 for d in best_scorer_results if not d['is_correct'])}")


def test_all_scorers(query: str, title_cache: List[tuple[int, str]], limit: int = 10):
    """Test all available rapidfuzz scorers and distance functions with and without preprocessing."""

    # Fuzz module scorers
    fuzz_scorers = [
        (fuzz.ratio, "fuzz.ratio", "Simple ratio comparison"),
        (fuzz.partial_ratio, "fuzz.partial_ratio", "Best partial string match"),
        (fuzz.token_sort_ratio, "fuzz.token_sort_ratio", "Ratio with sorted tokens"),
        (fuzz.token_set_ratio, "fuzz.token_set_ratio", "Ratio with token sets"),
        (fuzz.token_ratio, "fuzz.token_ratio", "Best of token_sort and token_set"),
        (fuzz.partial_token_sort_ratio, "fuzz.partial_token_sort_ratio", "Partial match with sorted tokens"),
        (fuzz.partial_token_set_ratio, "fuzz.partial_token_set_ratio", "Partial match with token sets"),
        (fuzz.partial_token_ratio, "fuzz.partial_token_ratio", "Best partial token match"),
        (fuzz.WRatio, "fuzz.WRatio", "Weighted ratio (auto-selects best method)"),
        (fuzz.QRatio, "fuzz.QRatio", "Quick ratio (faster, less accurate)"),
    ]

    # Custom scorer
    custom_scorers = [
        (token_aware_scorer, "custom.token_aware", "Token-aware scorer with length penalties for better accuracy"),
    ]

    # Distance module - these return distances (lower is better) so we need normalized versions
    # We'll use the similarity versions which return 0-100 like the fuzz scorers
    distance_scorers = [
        (distance.Levenshtein.normalized_similarity, "distance.Levenshtein", "Edit distance (insertions, deletions, substitutions)"),
        (distance.Indel.normalized_similarity, "distance.Indel", "Edit distance (only insertions and deletions)"),
        (distance.Hamming.normalized_similarity, "distance.Hamming", "Hamming distance (same length strings, substitutions only)"),
        (distance.Jaro.similarity, "distance.Jaro", "Jaro similarity (transpositions)"),
        (distance.JaroWinkler.similarity, "distance.JaroWinkler", "Jaro-Winkler similarity (prefix boost)"),
        (distance.LCSseq.normalized_similarity, "distance.LCSseq", "Longest Common Subsequence"),
        (distance.OSA.normalized_similarity, "distance.OSA", "Optimal String Alignment (Damerau-Levenshtein variant)"),
        (distance.DamerauLevenshtein.normalized_similarity, "distance.DamerauLevenshtein", "Damerau-Levenshtein distance (transpositions)"),
        (distance.Postfix.normalized_similarity, "distance.Postfix", "Postfix similarity"),
        (distance.Prefix.normalized_similarity, "distance.Prefix", "Prefix similarity"),
    ]

    all_scorers = custom_scorers + fuzz_scorers + distance_scorers

    # Test with and without preprocessing
    preprocessing_options = [
        (None, "No preprocessing"),
        (utils.default_process, "With preprocessing (default_process: lowercase, remove non-alphanumeric)")
    ]

    print(f"\nSearching for: '{query}'")
    print(f"Title cache size: {len(title_cache)} entries")
    print("=" * 120)

    for processor, processor_desc in preprocessing_options:
        print(f"\n{'=' * 120}")
        print(f"PREPROCESSING: {processor_desc}")
        print(f"{'=' * 120}")

        for scorer, name, description in all_scorers:
            print(f"\n### {name}")
            print(f"Description: {description}")
            print("-" * 120)

            try:
                start_time = time.time()
                results = search_titles_with_scorer(query, title_cache, scorer, name, limit=limit, processor=processor)
                search_time = time.time() - start_time

                print(f"Search time: {search_time * 1000:.2f}ms")

                if not results:
                    print("No matches found.")
                    continue

                print(f"\nTop {len(results)} matches:")
                for i, (_msg_id, title, score) in enumerate(results, 1):
                    print(f"  {i:2d}. [{score:6.2f}] {title[:90]}")

            except Exception as e:
                print(f"ERROR: {e}")

            print()


def get_credentials(cfgfile: Path, environment: str):
    """Load credentials from config file."""
    log(f"loading config from {cfgfile}")
    config = toml.load(cfgfile)

    try:
        return config["ongcode_bot"][environment]
    except KeyError:
        log(f"ERROR: no configuration for ongcode_bot.{environment} in credentials file")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test different rapidfuzz distance algorithms on ongcode titles"
    )

    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Search query to test with different algorithms (not needed for --benchmark mode)"
    )

    parser.add_argument(
        "--credentials-file", "-c",
        type=Path,
        default=None,
        help="file with discord credentials (for finding dbfile)"
    )

    parser.add_argument(
        "--environment", "--env",
        type=str,
        default="test",
        help="environment to use (for finding dbfile)"
    )

    parser.add_argument(
        "--dbfile",
        type=Path,
        default=None,
        help="database file to use (overrides environment default)"
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="number of results to show for each algorithm (default: 10)"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run benchmark mode: test with 'quensrich' and filter to scorers that find Queensryche"
    )

    parsed_args = parser.parse_args()

    if parsed_args.credentials_file is None:
        parsed_args.credentials_file = Path(__file__).parent / "credentials.toml"

    return parsed_args


def main():
    args = parse_args()

    # Determine database file
    if args.dbfile is None:
        # Use default naming convention like ongcodebot does
        args.dbfile = Path(__file__).parent / f"ongcode_{args.environment}.db"

    # Check if database exists
    if not args.dbfile.exists():
        log(f"ERROR: Database file {args.dbfile} does not exist")
        sys.exit(1)

    log(f"INFO: Using database file {args.dbfile}")

    # Initialize database
    initialize_db(args.dbfile)

    # Load title cache
    log("INFO: Loading title database into memory...")
    title_cache = load_title_cache()
    log(f"INFO: Loaded {len(title_cache)} titles into memory")

    if not title_cache:
        log("ERROR: No titles found in database")
        sys.exit(1)

    # Run appropriate mode
    if args.benchmark:
        benchmark_scorers(title_cache, limit=args.limit)
    else:
        if args.query is None:
            log("ERROR: query argument is required when not using --benchmark mode")
            sys.exit(1)
        test_all_scorers(args.query, title_cache, limit=args.limit)


if __name__ == "__main__":
    main()
