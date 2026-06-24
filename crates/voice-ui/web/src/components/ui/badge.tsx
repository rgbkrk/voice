import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        outline: "border-border text-foreground",
        converse: "border-amber-400/30 bg-amber-400/15 text-amber-300",
        play: "border-blue-300/30 bg-blue-300/15 text-blue-300",
        picked: "border-emerald-300/30 bg-emerald-300/15 text-emerald-300",
        waiting: "border-amber-300/30 bg-amber-300/15 text-amber-300",
        queued: "border-border text-muted-foreground",
        skipped: "border-border text-muted-foreground line-through opacity-70",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
