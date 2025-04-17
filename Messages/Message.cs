namespace AIProvider.Messages;

public record Message(string Content)
{
    public virtual string Role { get; set; } = "";
}