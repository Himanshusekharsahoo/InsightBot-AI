import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from '@/components/StatusBadge';

describe('StatusBadge', () => {
  it('renders completed status badge with green styling', () => {
    render(<StatusBadge status="completed" />);
    expect(screen.getByText('Completed')).toBeInTheDocument();
  });

  it('renders completed_unverified status badge with warning icon', () => {
    render(<StatusBadge status="completed_unverified" />);
    expect(screen.getByText('Completed (Unverified)')).toBeInTheDocument();
  });

  it('renders in_progress status with spinner', () => {
    render(<StatusBadge status="in_progress" />);
    expect(screen.getByText('In Progress')).toBeInTheDocument();
  });

  it('renders queued status', () => {
    render(<StatusBadge status="queued" />);
    expect(screen.getByText('Queued')).toBeInTheDocument();
  });

  it('renders provider_unavailable status with distinct badge', () => {
    render(<StatusBadge status="provider_unavailable" />);
    expect(screen.getByText('Provider Unavailable')).toBeInTheDocument();
  });

  it('renders failed status with error icon', () => {
    render(<StatusBadge status="failed" />);
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('renders cancelled status', () => {
    render(<StatusBadge status="cancelled" />);
    expect(screen.getByText('Cancelled')).toBeInTheDocument();
  });

  it('renders interrupted status badge with warning styling', () => {
    render(<StatusBadge status="interrupted" />);
    expect(screen.getByText('Interrupted')).toBeInTheDocument();
  });
});
