namespace AIProvider.Messages;

public record AssistantMessage(string Content) : Message(Content)
{
    public override string Role { get; set; } = "assistant";
}