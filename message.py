from enum import Enum
from typing import List, Union, Dict, Any, Optional
from PIL import Image
import hashlib
import re


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"


class ContentItem:
    """A single content item that can be text or image"""
    def __init__(self, content: Union[str, Image.Image], content_type: ContentType):
        self.content = content
        self.content_type = content_type
    
    def __str__(self) -> str:
        if self.content_type == ContentType.TEXT:
            return str(self.content)
        return "[IMAGE]"
    
    @classmethod
    def text(cls, text: str) -> "ContentItem":
        """Create a text content item"""
        return cls(text, ContentType.TEXT)
    
    @classmethod
    def image(cls, image: Image.Image) -> "ContentItem":
        """Create an image content item"""
        return cls(image, ContentType.IMAGE)

class MultimodalMessage:
    """A container for multimodal content (text and images) that preserves order and behaves like a message"""
    
    def __init__(self, content: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None):
        """
        Initialize with content which can be text, image, or a list of both.
        Order is preserved.
        
        Args:
            content: Initial content (text, image, or list of both)
        """
        self.items: List[ContentItem] = []
        self.additional_kwargs: Dict[str, Any] = {}
        
        if content is not None:
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        self.add_text(item)
                    elif isinstance(item, Image.Image):
                        self.add_image(item)
            elif isinstance(content, str):
                self.add_text(content)
            elif isinstance(content, Image.Image):
                self.add_image(content)
    
    @property
    def text(self) -> str:
        """Get all text content concatenated"""
        return "".join(str(item.content) for item in self.items if item.content_type == ContentType.TEXT)
    
    @property
    def images(self) -> List[Image.Image]:
        """Get all image content as a list"""
        return [item.content for item in self.items if item.content_type == ContentType.IMAGE]
    
    def __str__(self) -> str:
        """String representation with images represented as [IMAGE]"""
        return "".join(str(item) for item in self.items)
    
    def __add__(self, other: Union[str, "MultimodalMessage"]) -> "MultimodalMessage":
        """Support concatenation with strings and other MultimodalMessages"""
        result = MultimodalMessage()
        # Add all items from this message
        for item in self.items:
            if item.content_type == ContentType.TEXT:
                result.add_text(item.content)
            else:
                result.add_image(item.content)
        
        # Add items from other
        if isinstance(other, str):
            result.add_text(other)
        elif isinstance(other, MultimodalMessage):
            for item in other.items:
                if item.content_type == ContentType.TEXT:
                    result.add_text(item.content)
                else:
                    result.add_image(item.content)
        
        return result
    
    def __radd__(self, other: str) -> "MultimodalMessage":
        """Support right-side concatenation with strings"""
        if isinstance(other, str):
            result = MultimodalMessage()
            result.add_text(other)
            for item in self.items:
                if item.content_type == ContentType.TEXT:
                    result.add_text(item.content)
                else:
                    result.add_image(item.content)
            return result
        return NotImplemented
    
    def add_text(self, text: str) -> "MultimodalMessage":
        """Add text content, preserving order"""
        self.items.append(ContentItem.text(text))
        return self
    
    def add_image(self, image: Image.Image) -> "MultimodalMessage":
        """Add image content, preserving order"""
        self.items.append(ContentItem.image(image))
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation (similar to HumanMessage)"""
        # This could be expanded to match LangChain's format if needed
        return {
            "type": "multimodal",
            "items": [
                {"type": item.content_type.value, "content": item.content} 
                for item in self.items
            ],
            **self.additional_kwargs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalMessage":
        """Create from a dictionary representation"""
        message = cls()
        
        if "items" in data:
            for item_data in data["items"]:
                content = item_data["content"]
                if item_data["type"] == ContentType.TEXT.value:
                    message.add_text(content)
                elif item_data["type"] == ContentType.IMAGE.value:
                    message.add_image(content)
        
        # Handle additional kwargs
        for key, value in data.items():
            if key not in ["type", "items"]:
                message.additional_kwargs[key] = value
                
        return message

class MMDup:
    def __init__(self, *args):
        self.index_to_image: Dict[int, Image.Image] = {}
        self.hash_to_index: Dict[str, int] = {}
        for arg in args:
            if isinstance(arg, MultimodalMessage):
                for item in arg.items:
                    if item.content_type == ContentType.IMAGE:
                        self.register(item.content)
            elif isinstance(arg, Image.Image):
                self.register(arg)
            elif isinstance(arg, str):
                self.register(Image.open(arg))

    def hash_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.md5(buffer.getvalue()).hexdigest()

    def register(self, image: Image.Image, img_hash: str = None):
        img_hash = img_hash or self.hash_image(image)
        if img_hash not in self.hash_to_index:
            self.index_to_image[len(self.index_to_image) + 1] = image
            self.hash_to_index[img_hash] = len(self.index_to_image)

    def str_to_mm(self, text: str) -> MultimodalMessage:
        if isinstance(text, MultimodalMessage):
            return text
        assert isinstance(text, str)
        result = MultimodalMessage()
        last_end = 0
        for match in re.finditer(r'(<image_\d+>)', text):
            start, end = match.span()
            ref = match.group(1)
            if start > last_end:
                result.add_text(text[last_end:start])
            if ref in self.images_map:
                result.add_image(self.images_map[ref])
            else:
                raise
                result.add_text(ref)
            last_end = end
        if last_end < len(text):
            result.add_text(text[last_end:])
        return result

    def mm_to_str(self, mm: MultimodalMessage) -> str:
        if isinstance(mm, str):
            return mm
        assert isinstance(mm, MultimodalMessage)
        result = ""
        for item in mm.items:
            if item.content_type == ContentType.IMAGE:
                img_hash = self.hash_image(item.content)
                self.register(item.content, img_hash)
                result += f"<image_{self.hash_to_index[img_hash]}>"
            else:
                result += str(item.content)
        return result



def powerformat(_prompt: str, *args, **kwargs) -> Tuple[str, List[Image.Image]]:
    from .message import MultimodalMessage as MM, ContentType
    import copy
    def hash_image(img: Image.Image) -> str:
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return hashlib.md5(buffer.getvalue()).hexdigest()

    images: List[Image.Image] = []
    image_hash_to_index: Dict[str, int] = {}

    def process_arg(arg: Union[str, MM]):
        if isinstance(arg, MM):
            for item in arg.items:
                if item.content_type == ContentType.IMAGE:
                    img_hash = hash_image(item.content)
                    if img_hash not in image_hash_to_index:
                        images.append(item.content)
                        image_hash_to_index[img_hash] = len(images)
                    item.content = f"<image_{image_hash_to_index[img_hash]}>"
                    item.content_type = ContentType.TEXT

    args = [copy.deepcopy(arg) for arg in args]
    kwargs = {k: copy.deepcopy(v) for k, v in kwargs.items()}

    for arg in args:
        process_arg(arg)
    for arg in kwargs.values():
        process_arg(arg)

    formatted = _prompt.format(*args, **kwargs)
    return MM(*images, formatted) if images else formatted
