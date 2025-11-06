from bs4 import BeautifulSoup, NavigableString

def parse_reuters_sgm(path):
    with open(path, "r", encoding="latin-1") as f:
        soup = BeautifulSoup(f, "lxml")

    articles = []

    for r in soup.find_all("reuters"):
        newid = r.get("newid")
        date = r.date.get_text(strip=True) if r.date else None
        
        text_tag = r.find("text")   # <-- FIX HERE
        title = text_tag.find("title").get_text(strip=True) if text_tag and text_tag.find("title") else None
        dateline = text_tag.find("dateline").get_text(strip=True) if text_tag and text_tag.find("dateline") else None
        
        body_tag = None
        for t in ["body", "BODY", "Body"]:
            found = text_tag.find(t)
            if found:
                body_tag = found
                break

        if body_tag:
            body = body_tag.get_text(" ", strip=True)

        else:
            parts = []
            for child in text_tag.children:
                if isinstance(child, NavigableString):
                    txt = child.strip()
                    if txt:
                        parts.append(txt)

                elif child.name not in ["title", "dateline"]:
                    txt = child.get_text(" ", strip=True)
                    if txt:
                        parts.append(txt)

            body = " ".join(parts).strip()
        
        if "blah blah blah" in body.lower():
            continue
        fields = {
            "newid": newid,
            "date": date,
            "title": title,
            "dateline": dateline,
            "body": body,
        }

        item = ", ".join(f"{k}: {v}" for k, v in fields.items())
        articles.append(item)

    return articles


data = parse_reuters_sgm("reut2-002.sgm")

with open("docs/docs_002", "w", encoding="ascii") as f:
    f.write("\n".join([element.replace('\x02', '').replace('\x03', '') for element in data]))
