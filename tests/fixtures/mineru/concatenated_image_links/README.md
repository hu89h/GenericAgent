# MinerU concatenated image-link fixture

`full.md` is a reduced capture of the malformed MinerU output observed in the
paper import. Two consecutive local image targets were joined immediately
after the first `.jpg` suffix. The test materializes valid small JPEGs in its
temporary copy. Only the first path is present in the reduced tree, matching
the observed flattened MinerU output; the second path must be recovered from
the unique basename. This fixture covers the importer's path normalization
and copying contract, while image decoding is covered by the multimodal
tests.
