import { cn } from "@/lib/utils";

export function AgentAvatar({
  initial,
  color,
  size = "md",
  className,
}: {
  initial: string;
  color: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid shrink-0 place-items-center rounded-md font-bold text-neutral-950",
        size === "sm" && "h-9 w-9 text-xs",
        size === "md" && "h-11 w-11 text-sm",
        size === "lg" && "h-13 w-13 text-lg",
        className,
      )}
      style={{ backgroundColor: color }}
    >
      {initial}
    </div>
  );
}
