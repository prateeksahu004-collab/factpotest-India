from factprotest.rag.kb import Chunk, chunk_text, split_sections
from factprotest.rag.retriever import HybridRetriever, reciprocal_rank_fusion
from tests.fakes import HashEncoder


def test_split_sections_drops_reference_sections():
    text = "Lead para.\n\n== History ==\nHistory text.\n\n=== Early ===\nEarly text.\n\n== References ==\nRef list."
    sections = split_sections(text)
    assert [name for name, _ in sections] == ["Introduction", "History", "History › Early"]


def test_chunk_text_overlaps_and_respects_target():
    sentences = [f"Sentence number {i} talks about the protest in some detail here." for i in range(60)]
    chunks = chunk_text(" ".join(sentences), target_words=100, overlap_words=20)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 115 for c in chunks)
    first_tail = chunks[0].split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], k=60)
    assert fused[1] == fused[2] > fused[3] > 0
    assert set(fused) == {1, 2, 3, 4}


def _retriever() -> HybridRetriever:
    texts = {
        1: ["The farm laws were repealed in November 2021 after a year of protest.",
            "Farmers camped at Delhi borders demanding a legal guarantee for MSP."],
        2: ["The NEET-UG 2024 paper leak was investigated by the CBI.",
            "Grace marks were given to 1,563 candidates in NEET 2024."],
    }
    chunks = [Chunk(f"P{pid}-x-{i:03d}", pid, f"Article {pid}", "Intro", "https://example.org", t)
              for pid, ts in texts.items() for i, t in enumerate(ts)]
    encoder = HashEncoder()
    return HybridRetriever(chunks, encoder.documents([c.text for c in chunks]), encoder)


def test_hybrid_search_finds_lexical_match_and_filters_by_protest():
    r = _retriever()
    top = r.search("When were the farm laws repealed?", k=1, protest_id=1)[0]
    assert top.chunk.chunk_id == "P1-x-000"
    assert all(h.chunk.protest_id == 2 for h in r.search("paper leak CBI", k=2, protest_id=2))


def test_unknown_protest_falls_back_to_whole_kb():
    assert r_ids(_retriever().search("grace marks", k=1, protest_id=99)) == ["P2-x-001"]


def r_ids(hits):
    return [h.chunk.chunk_id for h in hits]
