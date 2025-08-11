namespace AIProvider;

public abstract partial record Provider
{
  // uses the openai compatibility layer of gemini
  public record GeminiProvider : OpenAiProvider
    {
        protected override string Key => "Gemini";
        protected override string Url => "https://generativelanguage.googleapis.com/v1beta/openai/";
    }

}
