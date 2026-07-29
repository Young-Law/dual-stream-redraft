*** Begin Patch
*** Update File: dualstream/compact_evidence.py
@@
-_HEADER_V33 = struct.Struct("<8sHBBQIHH32sI32sI B B H I I 32s I 32s B H H H H H H H H")
-_CHUNK_V33 = struct.Struct("<IIHHBBHHHHII32s")
-_TOKEN_V33_PREFIX = struct.Struct("<BBBB")
-_TOPK_V33 = struct.Struct("<IB")
-_SPAN_V33 = struct.Struct("<IIHBI")
-_SPAN_V33_EVAL = struct.Struct("<I")
-_MANIFEST_V33 = struct.Struct("<32s32s32sIII II IIII HHHHHH 32s B 32s")
+_HEADER_V33 = struct.Struct(
+    "<8sHBBBBQIIII32s32sI32sI32sIIIIHHBBBBIHHHH"
+)
+
+# chunk: index, first_token, token_count, flags, base_k, max_eff_k,
+# rank_count, stochastic_count, history_count, canary_count,
+# payload_len, crc32, payload_hash
+_CHUNK_V33 = struct.Struct("<IIHBBBBBBBBI I 32s")
+
+# token prefix: delta (uint8), chosen_rank (uint8), trigger_flags (uint8), record_flags (uint8)
+_TOKEN_V33_PREFIX = struct.Struct("<BBBB")
+_TOPK_V33 = struct.Struct("<IB")
+_SPAN_V33 = struct.Struct("<IIHBI")
+_SPAN_V33_EVAL = struct.Struct("<I")
+
+# manifest fields flattened: artifact_hash, header_hash, chunk_root, token_count, chunk_count,
+# span_count, raw_bytes, min_reconstructable, hist[4], trigger_counts[6], bitmap_hash, review_flag, retention_hash
+_MANIFEST_V33 = struct.Struct("<32s32s32sIII I I 4I 6I 32s B 32s")
*** End Patch
