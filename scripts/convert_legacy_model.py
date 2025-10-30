# Converts a legacy Keras h5 to modern TensorFlow 2
# This was kind of trial and error and may not work for all legacy h5 files

import json, os, sys, traceback
import h5py
import numpy as np

LEGACY_H5 = "../server/models/NewlyTrainedModel232025.h5"
OUT_KERAS = "sprite_fixed_model.keras"
OUT_SAVED = "sprite_fixed_savedmodel"
INPUT_SHAPE_FALLBACK = (256, 256, 3)


def _load_model_config(path):
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            raise RuntimeError("No model_config attribute in H5 (weights-only file?)")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw)


def _clean_config(cfg):
    # Remove batch_shape and fix dtype policies
    cfg = json.loads(json.dumps(cfg))
    if cfg.get("class_name") == "Sequential":
        layers = cfg.get("config", {}).get("layers", [])
    else:
        layers = cfg.get("config", {}).get("layers", [])

    def clean_layer(layer):
        lc = layer.get("config", {})
        # map batch_shape to input_shape
        if "batch_shape" in lc and "input_shape" not in lc:
            bs = lc.pop("batch_shape", None)
            if isinstance(bs, list) and len(bs) >= 2:
                lc["input_shape"] = bs[1:]
        # map batch_input_shape to input_shape
        if "batch_input_shape" in lc and "input_shape" not in lc:
            bis = lc.pop("batch_input_shape", None)
            if isinstance(bis, list) and len(bis) >= 2:
                lc["input_shape"] = bis[1:]
        # remove dtype policy objects
        if isinstance(lc.get("dtype"), dict) and lc["dtype"].get("class_name") in ("DTypePolicy","Policy"):
            lc["dtype"] = "float32"
        # remove unknown keys that seem to be messing up the deserializers
        for k in ["_name_scope", "graph_initializers"]:
            lc.pop(k, None)
        # normalize
        for key in list(lc.keys()):
            v = lc[key]
            if isinstance(v, dict) and "class_name" in v and "config" in v:
                # ensure there are no nested dtype policies
                if isinstance(v.get("config"), dict) and "dtype" in v["config"]:
                    if isinstance(v["config"]["dtype"], dict):
                        v["config"]["dtype"] = "float32"
        layer["config"] = lc

    for layer in layers:
        clean_layer(layer)
    return cfg


def _strategy_json_deserialize(clean_cfg):
    from tensorflow.keras.models import model_from_json
    j = json.dumps(clean_cfg)
    return model_from_json(j)


def _read_weights_index(path):
    mapping = {}
    with h5py.File(path, "r") as f:
        if "model_weights" not in f:
            # very old files may differ, try all groups
            groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        else:
            groups = ["model_weights"]

        for gname in groups:
            g = f[gname]
            for layer_name, lg in g.items():
                if not isinstance(lg, h5py.Group):
                    continue
                tensors = []
                # find stored weights
                for sub_name, sub in lg.items():
                    if isinstance(sub, h5py.Dataset):
                        tensors.append((sub_name, np.array(sub)))
                    elif isinstance(sub, h5py.Group):
                        for ds_name, ds in sub.items():
                            if isinstance(ds, h5py.Dataset):
                                tensors.append((f"{sub_name}/{ds_name}", np.array(ds)))
                if tensors:
                    mapping[layer_name] = tensors
    return mapping


def _strategy_layerwise_rebuild(clean_cfg, weights_map):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # build sequential
    assert clean_cfg.get("class_name") == "Sequential", "Only Sequential handled in this fallback"
    seq = models.Sequential(name=clean_cfg["config"].get("name","sequential"))

    # recreate layers from config
    supported = {
        "Conv2D", "MaxPooling2D", "AveragePooling2D",
        "Dropout", "Flatten", "Dense", "Activation", "BatchNormalization"
    }

    for layer in clean_cfg["config"].get("layers", []):
        lcls = layer["class_name"]
        lc   = layer.get("config", {})
        if lcls == "InputLayer":
            # sequential will create one input implicitly, but add explicit if present
            ishape = tuple(lc.get("batch_input_shape") or lc.get("input_shape") or INPUT_SHAPE_FALLBACK)
            if ishape and ishape[0] is None:
                ishape = ishape[1:]
            seq.add(layers.Input(shape=ishape))
            continue

        if lcls not in supported:
            # skip unknown layers
            print(f"Skipping unsupported layer class: {lcls}")
            continue

        # map args
        if lcls == "Conv2D":
            seq.add(layers.Conv2D(
                filters=lc.get("filters"),
                kernel_size=tuple(lc.get("kernel_size", (3,3))),
                strides=tuple(lc.get("strides", (1,1))),
                padding=lc.get("padding","valid"),
                activation=lc.get("activation", None),
                use_bias=lc.get("use_bias", True)
            ))
        elif lcls == "MaxPooling2D":
            seq.add(layers.MaxPooling2D(
                pool_size=tuple(lc.get("pool_size",(2,2))),
                strides=tuple(lc.get("strides",(2,2))),
                padding=lc.get("padding","valid")
            ))
        elif lcls == "AveragePooling2D":
            seq.add(layers.AveragePooling2D(
                pool_size=tuple(lc.get("pool_size",(2,2))),
                strides=tuple(lc.get("strides",(2,2))),
                padding=lc.get("padding","valid")
            ))
        elif lcls == "Flatten":
            seq.add(layers.Flatten())
        elif lcls == "Dropout":
            seq.add(layers.Dropout(rate=lc.get("rate",0.5)))
        elif lcls == "Dense":
            seq.add(layers.Dense(
                units=lc.get("units"),
                activation=lc.get("activation", None),
                use_bias=lc.get("use_bias", True)
            ))
        elif lcls == "Activation":
            seq.add(layers.Activation(lc.get("activation","linear")))
        elif lcls == "BatchNormalization":
            seq.add(layers.BatchNormalization(
                momentum=lc.get("momentum", 0.99),
                epsilon=lc.get("epsilon", 0.001),
                center=lc.get("center", True),
                scale=lc.get("scale", True)
            ))
        else:
            pass

    # assign weights by layer name if shapes match
    name_to_layer = {l.name: l for l in seq.layers if hasattr(l, "get_weights")}
    for lname, tensors in weights_map.items():
        if lname in name_to_layer:
            layer = name_to_layer[lname]
            # Keras ordering: typically [kernel, bias] or BN gamma/beta/moving_mean/moving_variance
            current = layer.get_weights()
            # build list by matching shapes
            assign = []
            for want in current:
                matched = None
                for _, arr in tensors:
                    if arr.shape == want.shape:
                        matched = arr
                        break
                if matched is None:
                    # skip assignment for this layer
                    assign = None
                    break
                assign.append(matched)
            if assign is not None and len(assign) == len(current):
                layer.set_weights(assign)

    return seq

def main():
    print("Reading legacy model_config")
    cfg = _load_model_config(LEGACY_H5)
    clean_cfg = _clean_config(cfg)

    try:
        print("Attempting JSON-based rebuild (strategy A)")
        m = _strategy_json_deserialize(clean_cfg)
        print("Strategy A succeeded!")
    except Exception as e:
        print("Strategy A failed, attempting layerwise reconstruction (strategy B).")
        try:
            weights_map = _read_weights_index(LEGACY_H5)
            m = _strategy_layerwise_rebuild(clean_cfg, weights_map)
            print("Strategy B succeeded!")
        except Exception as ee:
            traceback.print_exc()
            print("Both strategies failed.")
            sys.exit(2)

    # save modernized artifacts
    print(f"Saving modern Keras file: {OUT_KERAS}")
    m.save(OUT_KERAS)
    print(f"Exporting SavedModel dir: {OUT_SAVED}")
    try:
        m.export(OUT_SAVED)
    except Exception:
        try:
            import tensorflow as tf
            tf.saved_model.save(m, OUT_SAVED)
        except Exception:
            print("Could not export SavedModel with this TF/Keras combo .keras file is still usable.")
    print("Done.")


if __name__ == "__main__":
    main()
