import json
import requests # Assuming requests is used by CivitaiSearch

# This is the CivitaiSearch class provided by the user
# (Included for context, no changes made to this part)
class CivitaiSearch:
    BASE_URL = "https://search.civitai.com/multi-search"

    def __init__(self, token: str):
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,it;q=0.8",
            "authorization": f"Bearer {token}",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://civitai.com",
            "pragma": "no-cache",
            "referer": "https://civitai.com/",
            "user-agent": "Mozilla/5.0", # Standard User-Agent
            "x-meilisearch-client": "Meilisearch instant-meilisearch (v0.13.5) ; Meilisearch JavaScript (v0.34.0)"
        }

    def build_query(self, search_term="asd", limit=51, offset=0,
                    checkpoint_types=None,
                    file_formats=None,
                    model_types=None,
                    base_models=None,
                    nsfw_levels=None,
                    extra_filter=None):
        filter_list = []

        if checkpoint_types:
            filter_list.append([f"\"checkpointType\"=\"{t}\"" for t in checkpoint_types])
        if file_formats:
            filter_list.append([f"\"fileFormats\"=\"{f}\"" for f in file_formats])
        if model_types:
            filter_list.append([f"\"type\"=\"{t}\"" for t in model_types])
        if base_models:
            filter_list.append([f"\"version.baseModel\"=\"{b}\"" for b in base_models])
        
        nsfw_values = nsfw_levels if nsfw_levels else [1, 2, 4, 8, 16] # Default to all levels if None
        nsfw_filter_parts = [f"nsfwLevel={level}" for level in nsfw_values]
        
        # Ensure nsfw_filter_parts is not empty before joining
        if nsfw_filter_parts:
            nsfw_filter = " OR ".join(nsfw_filter_parts)
        else: # Fallback if nsfw_levels was an empty list, though default above should prevent this
            nsfw_filter = "nsfwLevel=1" # Or some other safe default

        base_filter = "(poi != true OR user.id = 58660) AND (availability != Private OR user.id = 58660)"
        full_extra_filter = f"{base_filter} AND ({nsfw_filter})"

        if extra_filter:
            full_extra_filter = f"{full_extra_filter} AND ({extra_filter})"

        filter_list.append(full_extra_filter)
        
        return {
            "queries": [
                {
                    "q": search_term,
                    "indexUid": "models_v9",
                    "facets": [
                        "category.name", "checkpointType", "fileFormats",
                        "lastVersionAtUnix", "tags.name", "type", "user.username",
                        "version.baseModel"
                    ],
                    "attributesToHighlight": [],
                    "highlightPreTag": "__ais-highlight__",
                    "highlightPostTag": "__/ais-highlight__",
                    "limit": limit,
                    "offset": offset,
                    "filter": filter_list
                }
            ]
        }

    def search(self, **kwargs):
        query = self.build_query(**kwargs)
        # print(json.dumps(query, indent=2)) # For debugging the query
        response = requests.post(self.BASE_URL, headers=self.headers, data=json.dumps(query))

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Errore {response.status_code}: {response.text}")

# --- Transformation Logic ---
BASE_CIVITAI_IMAGE_URL_PREFIX = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA"

def sanitize_filename_part(name_part):
    if not name_part: return ""
    # Basic sanitization for filenames
    return str(name_part).replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_') \
                         .replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_') \
                         .replace('>', '_').replace('|', '_')

def get_file_extension_from_format(format_string):
    if not format_string:
        return "bin" # Default extension
    f_lower = format_string.lower()
    if "safetensor" in f_lower:
        return "safetensors"
    if "pickletensor" in f_lower:
        return "ckpt"
    # Add more mappings as needed
    return f_lower # Fallback

def transform_search_hashdata_to_official_hashes(hash_data_list):
    hashes_dict = {}
    if isinstance(hash_data_list, list):
        for hd_entry in hash_data_list:
            if isinstance(hd_entry, dict) and "type" in hd_entry and "hash" in hd_entry:
                hashes_dict[hd_entry["type"]] = hd_entry["hash"]
    return hashes_dict

def adapt_search_response_to_official_api(search_response_json):
    """
    Transforms the Civitai search API response to a structure similar to the official GET /models API.
    """
    official_api_items = []

    if not search_response_json or "results" not in search_response_json or \
       not search_response_json["results"] or "hits" not in search_response_json["results"][0]:
        return {"items": []}

    search_hits = search_response_json["results"][0]["hits"]

    for hit in search_hits:
        # --- Model Level ---
        model_metrics = hit.get("metrics", {})
        model_permissions = hit.get("permissions", {})
        
        # Calculate model nsfwLevel: sum of flags from search result
        model_nsfw_level_flags = hit.get("nsfwLevel", [])
        if isinstance(model_nsfw_level_flags, list):
            calculated_model_nsfw_level = sum(model_nsfw_level_flags)
        else: # Should be a list based on example, but handle if it's not
            calculated_model_nsfw_level = 0


        official_item = {
            "id": hit.get("id"),
            "name": hit.get("name"),
            "description": "",  # Official API often has this empty for the model itself
            "allowNoCredit": model_permissions.get("allowNoCredit", False),
            "allowCommercialUse": model_permissions.get("allowCommercialUse", []),
            "allowDerivatives": model_permissions.get("allowDerivatives", True),
            "allowDifferentLicense": model_permissions.get("allowDifferentLicense", False),
            "type": hit.get("type"),
            "minor": hit.get("minor", False),
            "sfwOnly": hit.get("sfwOnly", False), # This key is present in the search example
            "poi": hit.get("poi", False),
            "nsfw": hit.get("nsfw", False),
            "nsfwLevel": calculated_model_nsfw_level,
            "availability": hit.get("availability", "Public"),
            "cosmetic": hit.get("cosmetic"),
            "supportsGeneration": hit.get("canGenerate", False), # Renamed from canGenerate
            "stats": {
                "downloadCount": model_metrics.get("downloadCount", 0),
                "favoriteCount": model_metrics.get("favoriteCount", 0), # Official API might show 0 here
                "thumbsUpCount": model_metrics.get("thumbsUpCount", 0),
                "thumbsDownCount": 0,  # Not in search model metrics; official example has it
                "commentCount": model_metrics.get("commentCount", 0),
                "ratingCount": model_metrics.get("ratingCount", 0), # Official API might show 0
                "rating": model_metrics.get("rating", 0.0),         # Official API might show 0
                "tippedAmountCount": model_metrics.get("tippedAmountCount", 0)
            },
            "tags": [tag.get("name") for tag in hit.get("tags", []) if tag.get("name")]
        }

        # --- Model Versions Level ---
        official_model_versions = []
        
        # Pre-group images by modelVersionId for easier lookup
        images_by_version_id = {}
        for img in hit.get("images", []):
            mv_id = img.get("modelVersionId")
            if mv_id:
                if mv_id not in images_by_version_id:
                    images_by_version_id[mv_id] = []
                images_by_version_id[mv_id].append(img)

        for index, search_version in enumerate(hit.get("versions", [])):
            version_id = search_version.get("id")
            version_metrics = search_version.get("metrics", {})

            official_version_data = {
                "id": version_id,
                "index": index,
                "name": search_version.get("name"),
                "baseModel": search_version.get("baseModel"),
                "baseModelType": search_version.get("baseModelType"),
                "publishedAt": search_version.get("createdAt"), # Search uses createdAt for version
                "availability": "Public",  # Default, not in search version object
                "nsfwLevel": search_version.get("nsfwLevel"), # This is per-version
                "description": search_version.get("description"), # Usually null/missing in search version
                "trainedWords": search_version.get("trainedWords", []),
                "stats": {
                    "downloadCount": version_metrics.get("downloadCount", 0),
                    "ratingCount": version_metrics.get("ratingCount", 0),
                    "rating": version_metrics.get("rating", 0.0),
                    "thumbsUpCount": version_metrics.get("thumbsUpCount", 0),
                    "thumbsDownCount": version_metrics.get("thumbsDownCount", 0) # Present in version metrics
                },
                "supportsGeneration": False,  # Official API example shows this as false for versions
                "files": [],
                "images": [],
                "downloadUrl": f"https://civitai.com/api/download/models/{version_id}"
            }

            # --- Files for Version (Simplified) ---
            # Search API doesn't provide detailed file info like official GET /model/:id
            # We'll create one primary file entry based on available info.
            model_file_formats = hit.get("fileFormats", [])
            version_hash_data = search_version.get("hashData", [])

            if model_file_formats and version_hash_data:
                primary_file_format_str = model_file_formats[0] # Assume first is primary
                file_extension = get_file_extension_from_format(primary_file_format_str)
                
                # Construct a filename
                sane_model_name = sanitize_filename_part(hit.get("name", "model"))
                sane_version_name = sanitize_filename_part(search_version.get("name", f"v{index + 1}"))
                # Example filename: modelname-versionname.ext
                constructed_filename = f"{sane_model_name}-{sane_version_name}.{file_extension}"

                file_entry = {
                    "id": version_id + 100000,  # Placeholder file ID, not from search
                    "sizeKB": None,  # Not available from search, official API has it
                    "name": constructed_filename,
                    "type": "Model",  # Default type
                    "pickleScanResult": "Success",  # Placeholder
                    "pickleScanMessage": "No Pickle imports",  # Placeholder
                    "virusScanResult": "Success",  # Placeholder
                    "virusScanMessage": None,  # Placeholder
                    "scannedAt": search_version.get("createdAt"),  # Placeholder
                    "metadata": {
                        "format": primary_file_format_str,
                        "size": "full",  # Placeholder
                        "fp": "fp16"  # Placeholder, common
                    },
                    "hashes": transform_search_hashdata_to_official_hashes(version_hash_data),
                    "downloadUrl": f"https://civitai.com/api/download/models/{version_id}", # Same as version for primary
                    "primary": True
                }
                official_version_data["files"].append(file_entry)

            # --- Images for Version ---
            version_images_list = images_by_version_id.get(version_id, [])
            for search_img in version_images_list:
                img_id = search_img.get("id")
                img_url_segment = search_img.get("url")
                img_width_for_url = search_img.get("width", 512) # Default if width missing

                constructed_img_url = None
                if img_id and img_url_segment:
                     # Official API image URLs use .jpeg, server handles format.
                    constructed_img_url = f"{BASE_CIVITAI_IMAGE_URL_PREFIX}/{img_url_segment}/width={img_width_for_url}/{img_id}.jpeg"
                
                image_entry = {
                    "id": img_id,
                    "url": constructed_img_url,
                    "nsfwLevel": search_img.get("nsfwLevel", 0),
                    "width": search_img.get("width"),
                    "height": search_img.get("height"),
                    "hash": search_img.get("hash"),
                    "type": search_img.get("type", "image"),
                    "minor": search_img.get("minor", False),
                    "poi": search_img.get("poi", False),
                    "hasMeta": search_img.get("hasMeta", False),
                    "hasPositivePrompt": search_img.get("hasPositivePrompt", False),
                    "onSite": search_img.get("onSite", False),
                    "remixOfId": search_img.get("remixOfId")
                }
                official_version_data["images"].append(image_entry)
            
            official_model_versions.append(official_version_data)

        official_item["modelVersions"] = official_model_versions
        official_api_items.append(official_item)

    return {"items": official_api_items}


# Example of how to use the transformation (assuming you have `results_from_search`)
if __name__ == "__main__":
    # This is the example usage from your code
    token = "4c7745e54e872213201291ba1cae1aaca702941f291432cf4fef22803333e487" # Example token
    civitai_searcher = CivitaiSearch(token)
    
    print("Performing search (this will make a live API call)...")
    try:
        search_api_response = civitai_searcher.search(
            search_term="hentai",
            model_types=["Checkpoint"],
            base_models=["SDXL 1.0"],
            nsfw_levels=[1, 2, 4, 8, 16] # Explicitly pass all desired NSFW levels
        )
        
        print("\nOriginal Search API Response (first hit preview):")
        if search_api_response and search_api_response["results"] and search_api_response["results"][0]["hits"]:
            print(json.dumps(search_api_response["results"][0]["hits"][0], indent=2, ensure_ascii=False))
        else:
            print("No hits found or unexpected response structure.")

        print("\n--- Transforming to Official API Format ---")
        adapted_response = adapt_search_response_to_official_api(search_api_response)
        
        print("\nAdapted Response (first item preview):")
        if adapted_response and adapted_response["items"]:
             print(json.dumps(adapted_response["items"][0], indent=2, ensure_ascii=False))
        else:
            print("No items after transformation or empty result.")
        
        # To see the full adapted response:
        # print("\nFull Adapted Response:")
        # print(json.dumps(adapted_response, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"An error occurred: {e}")
    
    # You can also test with the provided sample JSON if you don't want to make a live call
    # sample_search_response_text = """ PASTE YOUR FULL SEARCH RESPONSE JSON HERE """
    # if sample_search_response_text:
    #     sample_search_response_json = json.loads(sample_search_response_text)
    #     adapted_from_sample = adapt_search_response_to_official_api(sample_search_response_json)
    #     print("\nAdapted Response from Sample JSON (first item):")
    #     if adapted_from_sample and adapted_from_sample["items"]:
    #         print(json.dumps(adapted_from_sample["items"][0], indent=2, ensure_ascii=False))