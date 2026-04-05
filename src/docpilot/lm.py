import dspy
from typing import Optional, Generator, AsyncGenerator, Any

class DspyLMWrapper(dspy.BaseLM):
    """Custom LM wrapper without DSPy signatures."""

    def __init__(self, model: str, **kwargs):
        # Initialize with BaseLM's parameters
        super().__init__(model=model, **kwargs)

        # Import litellm here to avoid circular imports
        import litellm
        self.litellm = litellm

    def _render_template(
            self,
            prompt: Optional[str] = None,
            template: Optional[str] = None,
            variables: Optional[dict[str, Any]] = None
        ) -> str | None:

        if variables is None:
            variables = {}

        if template:
            try:
                return template.format(**variables)
            except KeyError as e:
                raise ValueError(f"Missing template variable: {e}")

        return prompt

    def forward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        template: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> str | None:
        """Implement the forward method for non-streaming calls."""
        kwargs.pop('stream')
        final_prompt = self._render_template(prompt, template, variables)
        messages = messages or [{"role": "user", "content": final_prompt}]

        response = self.litellm.completion(
            model=self.model,
            messages=messages,
            **{**self.kwargs, **kwargs}
        )
        return response.choices[0].message.content

    async def aforward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        template: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> str | None:
        """Async version of forward."""
        kwargs.pop('stream')
        final_prompt = self._render_template(prompt, template, variables)
        messages = messages or [{"role": "user", "content": final_prompt}]

        response = await self.litellm.acompletion(
            model=self.model,
            messages=messages,
            **{**self.kwargs, **kwargs}
        )

        return response.choices[0].message.content

    def stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        template: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream responses without DSPy's streaming infrastructure."""
        kwargs.pop('stream')
        final_prompt = self._render_template(prompt, template, variables)
        messages = messages or [{"role": "user", "content": final_prompt}]

        response = self.litellm.completion(
            model=self.model,
            messages=messages,
            stream=True,
            **{**self.kwargs, **kwargs}
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def astream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        template: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async version of stream."""
        kwargs.pop('stream')
        final_prompt = self._render_template(prompt, template, variables)
        messages = messages or [{"role": "user", "content": final_prompt}]

        response = await self.litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=True,
            **{**self.kwargs, **kwargs}
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        template: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        if stream:
            return self.stream(prompt, messages, template, variables, **kwargs)
        else:
            return self.forward(prompt, messages, template, variables, **kwargs)

    async def __acall__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        template: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        if stream:
            return self.astream(prompt, messages, template, variables, **kwargs)
        else:
            return await self.aforward(prompt, messages, template, variables, **kwargs)

