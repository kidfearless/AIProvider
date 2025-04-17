namespace AIProvider.Messages;

public record SystemPromptMessage(string Content) : Message(Content)
{
    public override string Role { get; set; } = "system";
}