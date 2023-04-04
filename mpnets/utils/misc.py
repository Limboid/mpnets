from exports import export
import glom


@export
def is_valid_glom_string(obj, glom_str):
    try:
        glom.glom(obj, glom_str)
        return True
    except:
        return False
