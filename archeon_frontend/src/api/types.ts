export const JobStatus = {
    QUEUED: "queued",
    PROCESSING: "processing",
    COMPLETED: "completed",
    FAILED: "failed",
    CANCELLED: "cancelled"
} as const;

export type JobStatusType = typeof JobStatus[keyof typeof JobStatus];

export type JobType = 'text_to_3d' | 'image_to_3d';

export type MeshOpsAction = 'decimate' | 'convert';

export interface MeshOpsRequest {
    job_uid: string;
    action: MeshOpsAction;
    format?: 'glb' | 'obj' | 'ply' | 'stl';
    ratio?: number;
}

export interface BaseGenerationRequest {
    seed?: number;
    steps?: number;
    guidance?: number;
    octree_resolution?: number;
    format?: 'glb' | 'obj' | 'ply' | 'stl';
    texture?: boolean;
}

export interface TextTo3DRequest extends BaseGenerationRequest {
    type: 'text_to_3d';
    prompt: string;
}

export interface ImageTo3DRequest extends BaseGenerationRequest {
    type: 'image_to_3d';
    image: string; // Base64
    remove_background?: boolean;
}

export type JobRequest = TextTo3DRequest | ImageTo3DRequest;

export interface JobResponse {
    uid: string;
    status: JobStatusType;
    created_at: string;
    completed_at?: string;
    error?: string;
    file_path?: string;
}

export interface SystemMetrics {
    uptime: number;
    cpu_percent: number;
    memory: {
        total: number;
        available: number;
        percent: number;
        used: number
    };
    gpu?: {
        name: string;
        total_memory: number;
        free_memory: number;
        used_memory: number;
        utilization: number;
    };
}
