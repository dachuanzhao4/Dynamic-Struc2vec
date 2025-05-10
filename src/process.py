import math
import random
import matplotlib.pyplot as plt
import re
import collections
import os
import sys

def get_year_bin(year):
    if year < 1980:
        return "<1980"
    elif year <= 1984:
        return "1980-1984"
    elif year <= 1989:
        return "1985-1989"
    elif year <= 1994:
        return "1990-1994"
    elif year <= 1999:
        return "1995-1999"
    elif year <= 2004:
        return "2000-2004"
    elif year <= 2009:
        return "2005-2009"
    return None 

def balanced(input_file, output_file):

    bins = {
        "<1980": [],
        "1980-1984": [],
        "1985-1989": [],
        "1990-1994": [],
        "1995-1999": [],
        "2000-2004": [],
        "2005-2009": []
    }

    def flush_paper(p):
        if not p["year"] or not p["title"]:
            return
        try:
            y = int(p["year"])
        except:
            return
        bin_label = get_year_bin(y)
        if bin_label is None:
            return
        bins[bin_label].append(dict(
            title=p["title"],
            authors=p["authors"],
            year=p["year"],
            index=p["index"],
            lines=p["lines"][:]
        ))

    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
        current_paper = {
            "title": None,
            "authors": None,
            "year": None,
            "index": None,
            "lines": []
        }
        for line in fin:
            line_strip = line.strip()
            if line_strip.startswith("#*"):
                if current_paper["title"] or current_paper["authors"] or current_paper["year"]:
                    flush_paper(current_paper)
                current_paper = {
                    "title": line_strip[2:].strip(),
                    "authors": None,
                    "year": None,
                    "index": None,
                    "lines": [line]
                }
            elif line_strip.startswith("#@"):
                current_paper["authors"] = line_strip[2:].strip()
                current_paper["lines"].append(line)
            elif line_strip.startswith("#t"):
                current_paper["year"] = line_strip[2:].strip()
                current_paper["lines"].append(line)
            elif line_strip.startswith("#index"):
                current_paper["index"] = line_strip[6:].strip()
                current_paper["lines"].append(line)
            else:
                current_paper["lines"].append(line)
        if current_paper["title"] or current_paper["authors"] or current_paper["year"]:
            flush_paper(current_paper)

    total_papers = sum(len(bins[b]) for b in bins)
    if total_papers == 0:
        print("No valid papers found (<2010). Exiting.")
        return

    target = math.ceil(0.01 * total_papers)  
    used_bins = [b for b in bins if len(bins[b])>0]
    if len(used_bins) == 0:
        print("All bins empty??")
        return

    bin0_label = "<1980"
    bin0_nonempty = (bin0_label in used_bins and len(bins[bin0_label])>0)

    selected_papers = []
    if bin0_nonempty:
        k = len(used_bins)
        x = math.floor(target / (k+2))
        if x<=0:
            share_each = math.floor(target / k)
            print(f"Note: x=0 => fallback share_each={share_each}")
            for b in used_bins:
                bin_list = bins[b]
                bin_size = len(bin_list)
                if bin_size <= share_each:
                    selected_papers.extend(bin_list)
                else:
                    picked = random.sample(bin_list, share_each)
                    selected_papers.extend(picked)
        else:
            bin0_list = bins[bin0_label]
            share0 = 3*x
            share_others = x
            if len(bin0_list) <= share0:
                selected_papers.extend(bin0_list)
            else:
                sampled = random.sample(bin0_list, share0)
                selected_papers.extend(sampled)
            for b in used_bins:
                if b==bin0_label:
                    continue
                blist = bins[b]
                if len(blist)<= share_others:
                    selected_papers.extend(blist)
                else:
                    sampled = random.sample(blist, share_others)
                    selected_papers.extend(sampled)
    else:
        k = len(used_bins)
        share_each = math.floor(target/k)
        for b in used_bins:
            bin_list = bins[b]
            bin_size = len(bin_list)
            if bin_size <= share_each:
                selected_papers.extend(bin_list)
            else:
                picked = random.sample(bin_list, share_each)
                selected_papers.extend(picked)

    print(f"=== Summary ===")
    print(f"Total bins used: {len(used_bins)}, total papers={total_papers}, target={target}")
    if bin0_nonempty:
        print(f'  bin0=<1980 was non-empty => used triple strategy')
    print(f"Selected {len(selected_papers)} papers in total.")

    with open(output_file, "w", encoding="utf-8") as fout:
        for p in selected_papers:
            for line in p["lines"]:
                fout.write(line)
            fout.write("\n")
    print(f"Done. Wrote ~1% sample to {output_file}.")

def plot_year_distribution(dblp_file):
    bins = {
        "<1980": 0,
        "1980-1984": 0,
        "1985-1989": 0,
        "1990-1994": 0,
        "1995-1999": 0,
        "2000-2004": 0,
        "2005-2009": 0,
        ">=2010": 0
    }
    year_pattern = re.compile(r'^#t\s*(\d{4})\s*$')

    with open(dblp_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            match = year_pattern.match(line)
            if match:
                year_str = match.group(1)
                try:
                    year = int(year_str)
                except:
                    continue 
                if year < 1980:
                    bins["<1980"] += 1
                elif 1980 <= year <= 1984:
                    bins["1980-1984"] += 1
                elif 1985 <= year <= 1989:
                    bins["1985-1989"] += 1
                elif 1990 <= year <= 1994:
                    bins["1990-1994"] += 1
                elif 1995 <= year <= 1999:
                    bins["1995-1999"] += 1
                elif 2000 <= year <= 2004:
                    bins["2000-2004"] += 1
                elif 2005 <= year <= 2009:
                    bins["2005-2009"] += 1
                else:
                    bins[">=2010"] += 1

    bin_labels = list(bins.keys())
    counts = [bins[label] for label in bin_labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, counts, color="skyblue")
    plt.xlabel("Year Range")
    plt.ylabel("Number of Papers")
    plt.title("Year Distribution in 8 Bins")
    plt.savefig("graph/distribute.png")
    plt.close()


def split(input_file):
    output_files = {
        "1980.txt": open(os.path.join("graph", "1980.txt"), "w", encoding="utf-8"),
        "1985.txt": open(os.path.join("graph", "1985.txt"), "w", encoding="utf-8"),
        "1990.txt": open(os.path.join("graph", "1990.txt"), "w", encoding="utf-8"),
        "1995.txt": open(os.path.join("graph", "1995.txt"), "w", encoding="utf-8"),
        "2000.txt": open(os.path.join("graph", "2000.txt"), "w", encoding="utf-8"),
        "2005.txt": open(os.path.join("graph", "2005.txt"), "w", encoding="utf-8"),
        "2010.txt": open(os.path.join("graph", "2010.txt"), "w", encoding="utf-8"),
    }

    current_title = None
    current_authors = None
    current_year = None
    current_index = None

    def write_paper(title, authors, year, idx):
        try:
            y = int(year)
        except:
            return 

        if y >= 2010:
            return  

        if y < 1980:
            f_out = output_files["1980.txt"]
            f_out.write(f"#*{title}\n")
            f_out.write(f"#@{authors}\n")
            f_out.write(f"#t{year}\n")
            f_out.write(f"#index{idx}\n\n")

        if y < 1985:
            f_out = output_files["1985.txt"]
            f_out.write(f"#*{title}\n")
            f_out.write(f"#@{authors}\n")
            f_out.write(f"#t{year}\n")
            f_out.write(f"#index{idx}\n\n")

        if y < 1990:
            f_out = output_files["1990.txt"]
            f_out.write(f"#*{title}\n")
            f_out.write(f"#@{authors}\n")
            f_out.write(f"#t{year}\n")
            f_out.write(f"#index{idx}\n\n")

        if y < 1995:
            f_out = output_files["1995.txt"]
            f_out.write(f"#*{title}\n")
            f_out.write(f"#@{authors}\n")
            f_out.write(f"#t{year}\n")
            f_out.write(f"#index{idx}\n\n")

        if y < 2000:
            f_out = output_files["2000.txt"]
            f_out.write(f"#*{title}\n")
            f_out.write(f"#@{authors}\n")
            f_out.write(f"#t{year}\n")
            f_out.write(f"#index{idx}\n\n")

        if y < 2005:
            f_out = output_files["2005.txt"]
            f_out.write(f"#*{title}\n")
            f_out.write(f"#@{authors}\n")
            f_out.write(f"#t{year}\n")
            f_out.write(f"#index{idx}\n\n")

        if y < 2010:
            f_out = output_files["2010.txt"]
            f_out.write(f"#*{title}\n")
            f_out.write(f"#@{authors}\n")
            f_out.write(f"#t{year}\n")
            f_out.write(f"#index{idx}\n\n")


    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.rstrip("\n")

            if line.startswith("#*"):
                if current_title and current_authors and current_year and current_index:
                    write_paper(current_title, current_authors, current_year, current_index)

                current_title = line[2:].strip() 
                current_authors = None
                current_year = None
                current_index = None

            elif line.startswith("#@"):
                current_authors = line[2:].strip()

            elif line.startswith("#t"):
                current_year = line[2:].strip()

            elif line.startswith("#index"):
                current_index = line[6:].strip()

        if current_title and current_authors and current_year and current_index:
            write_paper(current_title, current_authors, current_year, current_index)

    for f in output_files.values():
        f.close()

def build_global_label(input_file, label_file):

    author_to_id = {}
    next_id = 0

    def add_author(author_name):
        nonlocal next_id
        author_name = author_name.strip()
        if author_name and (author_name not in author_to_id):
            author_to_id[author_name] = next_id
            next_id += 1

    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line_strip = line.strip()
            if line_strip.startswith("#@"):
                authors_str = line_strip[2:].strip()
                authors = authors_str.split(",")
                for a in authors:
                    a = a.strip()
                    if a:
                        add_author(a)

    with open(label_file, "w", encoding="utf-8") as fout:
        for author, aid in author_to_id.items():
            fout.write(f"{aid} {author}\n")

    print(f"build_global_label done. Found {len(author_to_id)} authors. Wrote label to {label_file}")



def create_edgelist_with_global_label(input_file, label_file, output_edgelist):
    
    if output_edgelist.endswith(".txt"):
        output_edgelist = output_edgelist[:-4] + ".edgelist"
    author_to_id = {}
    with open(label_file, "r", encoding="utf-8") as f_label:
        for line in f_label:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                aid_str, author_name = parts
                try:
                    aid = int(aid_str)
                    author_to_id[author_name] = aid
                except:
                    pass 

    with open(output_edgelist, "w", encoding="utf-8") as f_out:

        def write_pairs_for_paper(authors_str):
            authors_list = [a.strip() for a in authors_str.split(",") if a.strip()]
            ids = []
            for a in authors_list:
                if a in author_to_id:
                    ids.append(author_to_id[a])
                else:
                    pass
            n = len(ids)
            for i in range(n):
                for j in range(i + 1, n):
                    f_out.write(f"{ids[i]} {ids[j]}\n")

        with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
            current_authors = None

            for line in fin:
                line_strip = line.strip()
                if line_strip.startswith("#*"):
                    if current_authors:
                        write_pairs_for_paper(current_authors)
                    current_authors = None

                elif line_strip.startswith("#@"):
                    current_authors = line_strip[2:].strip()

            # flush last
            if current_authors:
                write_pairs_for_paper(current_authors)

    print(f"create_edgelist_with_global_label done. Wrote edges to {output_edgelist}")


def main():
    input_file = "graph/outputacm.txt"  

    balanced(input_file, "graph/dblp_small.txt")
    plot_year_distribution("graph/dblp_small.txt")
    split("graph/dblp_small.txt")
    build_global_label("graph/2010.txt", "graph/author_label")
    create_edgelist_with_global_label("graph/2010.txt", "graph/author_label", "graph/edgelist2010.txt")
    create_edgelist_with_global_label("graph/2005.txt", "graph/author_label", "graph/edgelist2005.txt")
    create_edgelist_with_global_label("graph/2000.txt", "graph/author_label", "graph/edgelist2000.txt")
    create_edgelist_with_global_label("graph/1995.txt", "graph/author_label", "graph/edgelist1995.txt")
    create_edgelist_with_global_label("graph/1990.txt", "graph/author_label", "graph/edgelist1990.txt")
    create_edgelist_with_global_label("graph/1985.txt", "graph/author_label", "graph/edgelist1985.txt")
    create_edgelist_with_global_label("graph/1980.txt", "graph/author_label", "graph/edgelist1980.txt")


if __name__ == "__main__":
    main()

