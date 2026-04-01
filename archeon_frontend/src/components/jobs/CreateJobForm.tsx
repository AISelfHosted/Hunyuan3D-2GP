import React, { useState } from 'react';
import { apiClient } from '../../api/client';
import type { JobRequest, JobType } from '../../api/types';
import { Wand2, Image as ImageIcon, Loader2 } from 'lucide-react';

export const CreateJobForm: React.FC = () => {
    const [mode, setMode] = useState<JobType>('text_to_3d');
    const [prompt, setPrompt] = useState('');
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

    // Advanced settings
    const [steps, setSteps] = useState(50);
    const [guidance, setGuidance] = useState(5.0);
    const [seed, setSeed] = useState(1234);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setSelectedImage(file);
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    const convertToBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = error => reject(error);
        });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSubmitting(true);
        setMessage(null);

        try {
            let request: JobRequest;

            if (mode === 'text_to_3d') {
                request = {
                    type: 'text_to_3d',
                    prompt,
                    steps,
                    guidance,
                    seed
                };
            } else if (mode === 'image_to_3d') {
                if (!selectedImage) throw new Error("Please select an image");
                const base64Image = await convertToBase64(selectedImage);
                request = {
                    type: 'image_to_3d',
                    image: base64Image,
                    steps,
                    guidance,
                    seed,
                    remove_background: true
                };
            } else {
                return;
            }

            await apiClient.post('/jobs', request);
            setMessage({ type: 'success', text: 'Job submitted successfully!' });
            // Ideally trigger refresh of job list
        } catch (err: any) {
            console.error(err);
            setMessage({ type: 'error', text: err.response?.data?.detail || err.message || 'Submission failed' });
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="bg-archeon-panel p-6 rounded-lg border border-gray-700">
            <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <Wand2 className="text-archeon-primary" /> NEW GENERATION
            </h2>

            {/* Mode Switcher */}
            <div className="flex gap-2 mb-6 bg-gray-900 p-1 rounded-lg">
                <button
                    onClick={() => setMode('text_to_3d')}
                    className={`flex-1 py-2 text-sm rounded-md transition-colors ${mode === 'text_to_3d' ? 'bg-archeon-primary text-white' : 'text-gray-400 hover:text-white'}`}
                >
                    Text to 3D
                </button>
                <button
                    onClick={() => setMode('image_to_3d')}
                    className={`flex-1 py-2 text-sm rounded-md transition-colors ${mode === 'image_to_3d' ? 'bg-archeon-primary text-white' : 'text-gray-400 hover:text-white'}`}
                >
                    Image to 3D
                </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
                {/* Inputs */}
                {mode === 'text_to_3d' ? (
                    <div>
                        <label className="block text-sm text-gray-400 mb-2">Prompt</label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full bg-gray-800 border border-gray-700 rounded p-3 text-sm focus:border-archeon-primary outline-none"
                            placeholder="A futuristic cyberpunk helmet..."
                            rows={3}
                            required
                        />
                    </div>
                ) : (
                    <div>
                        <label className="block text-sm text-gray-400 mb-2">Input Image</label>
                        <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center hover:border-archeon-primary transition-colors cursor-pointer relative">
                            <input
                                type="file"
                                onChange={handleImageChange}
                                accept="image/*"
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            />
                            {previewUrl ? (
                                <img src={previewUrl} alt="Preview" className="max-h-48 mx-auto rounded" />
                            ) : (
                                <div className="text-gray-500 flex flex-col items-center">
                                    <ImageIcon size={32} className="mb-2" />
                                    <span>Click or Drag to Upload</span>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Parameters (Condensed) */}
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs text-gray-500 mb-1">Steps ({steps})</label>
                        <input
                            type="range" min="10" max="100" value={steps} onChange={(e) => setSteps(Number(e.target.value))}
                            className="w-full accent-archeon-primary"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-gray-500 mb-1">Guidance ({guidance})</label>
                        <input
                            type="range" min="1" max="20" step="0.5" value={guidance} onChange={(e) => setGuidance(Number(e.target.value))}
                            className="w-full accent-archeon-primary"
                        />
                        <input
                            type="range" min="1" max="20" step="0.5" value={guidance} onChange={(e) => setGuidance(Number(e.target.value))}
                            className="w-full accent-archeon-primary"
                        />
                    </div>
                </div>

                {/* Seed */}
                <div>
                    <label className="block text-xs text-gray-500 mb-1">Seed</label>
                    <input
                        type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                        className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-2 text-sm focus:border-archeon-primary outline-none"
                    />
                </div>

                {/* Submit */}
                <button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full bg-linear-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold py-3 rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isSubmitting ? (
                        <><Loader2 className="animate-spin" size={20} /> Generating...</>
                    ) : (
                        "Generate Asset"
                    )}
                </button>

                {/* Feedback */}
                {message && (
                    <div className={`text-sm p-3 rounded ${message.type === 'success' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                        {message.text}
                    </div>
                )}
            </form>
        </div>
    );
};
