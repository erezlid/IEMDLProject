import json

path = './val.json'

with open(path, 'r') as f:
    val = json.load(f)
with open('./caption_outputs/g_norm_inverse/gaussian_norm_inverse_noise.json','r') as f:
    g = json.load(f)


g_norm_ids = [entry['image_id'] for entry in g]

val_entries = [entry for entry in val if entry['image_id'] in g_norm_ids]

with open('val_filtered.json','w') as f:
    json.dump(val_entries, f)




